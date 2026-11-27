import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'
import { api } from '../api/client'
import type { AiStatus, Annotation, BoxGeom, Detection, ImageInfo, Label, Mode, PolygonGeom, SimilarResult } from '../api/types'
import { AiSuggestionsToast, AiWorking } from '../components/AiActivity'
import { AnnotationCanvas } from '../components/AnnotationCanvas'
import { AnnotationList } from '../components/AnnotationList'
import { NamePicker } from '../components/NamePicker'
import { RemoveAiButton } from '../components/RemoveAiButton'
import { SimilarSearch, type SimilarState } from '../components/SimilarSearch'
import { TextSearch } from '../components/TextSearch'
import { ProposalPanel } from '../components/ProposalPanel'
import { Toolbar, TOOLS } from '../components/Toolbar'
import { HelpButton, Tour, useTour, type TourStep } from '../components/Tour'
import { effort } from '../lib/effort'
import { boxIou, boundingBox, nextColor, uid } from '../lib/geometry'
import { isAiCleared, setAiCleared } from '../lib/aiCleared'
import { loadSettings, saveSetting } from '../lib/settings'
import { btnPrimary, btnSecondary, statusStyle } from '../lib/ui'
import { useAnnotate, type Proposal, type Tool } from '../store/annotate'

type SaveState = 'saved' | 'saving' | 'error'

/** Detector boxes at or above this score are added without asking (when auto-accept is on). */
const AUTO_ACCEPT = 0.5
/** Minimum detector score used to name a shape the user drew. */
const NAME_MIN_SCORE = 0.25

export function AnnotatePage() {
  const params = useParams()
  const projectId = Number(params.projectId)
  const imageId = Number(params.imageId)
  const navigate = useNavigate()

  const [labels, setLabels] = useState<Label[]>([])
  const labelsRef = useRef<Label[]>([])
  const [images, setImages] = useState<ImageInfo[]>([])
  const [image, setImage] = useState<ImageInfo | null>(null)
  const [aiStatus, setAiStatus] = useState<AiStatus | null>(null)
  const [mode, setMode] = useState<Mode>(() => loadSettings(projectId).mode)
  const [saveState, setSaveState] = useState<SaveState>('saved')
  const [error, setError] = useState<string | null>(null)
  const [embedState, setEmbedState] = useState<'idle' | 'running' | 'ready' | 'failed'>('idle')
  const [smartBusy, setSmartBusy] = useState(false)
  const [detecting, setDetecting] = useState(false)
  const [autoSuggest, setAutoSuggest] = useState(() => loadSettings(projectId).autoSuggest)
  const [autoAccept, setAutoAccept] = useState(() => loadSettings(projectId).autoAccept)
  const [expert, setExpert] = useState(() => loadSettings(projectId).expert)
  const [suggested, setSuggested] = useState({ added: 0, review: 0, nonce: 0 })
  /** New shape waiting for the user to name it (the 'What is this?' card). */
  const [pending, setPending] = useState<{ draft: Annotation; tool: string } | null>(null)
  /** Every detection for the open image (also low-score ones), used to name drawn shapes. */
  const detectionsRef = useRef<Detection[]>([])
  /** "Find alike objects": the example shape, and the search card state. */
  const exampleRef = useRef<{ imageId: number; annotation: Annotation } | null>(null)
  const [similar, setSimilar] = useState<SimilarState | null>(null)
  const similarStop = useRef(false)
  const similarIds = useRef<string[]>([])

  const store = useAnnotate()
  const { version, tool, activeLabelId, smartPoints, smartBox } = store
  const imageIdRef = useRef<number | null>(null)
  const savedVersion = useRef(0)
  const segmentSeq = useRef(0)

  const tour = useTour('annotator-tour-annotate-v1', !!image)

  const samReady = mode === 'ai' && !!aiStatus?.sam.available
  const yoloReady = mode === 'ai' && !!aiStatus?.yolo.available
  /** OWLv2 present: "find alike objects" (mark one, the AI marks every object of that kind). */
  const alikeReady = mode === 'ai' && !!aiStatus?.owl?.available
  /** OWLv2 text tower present: "find by name" (type "car", the AI marks every car). */
  const textReady = mode === 'ai' && !!aiStatus?.owl_text?.available

  // ---- project data ------------------------------------------------------
  const setLabelList = (list: Label[]) => {
    labelsRef.current = list
    setLabels(list)
  }

  useEffect(() => {
    Promise.all([api.listLabels(projectId), api.listImages(projectId)])
      .then(([l, i]) => {
        setLabelList(l)
        setImages(i)
        if (useAnnotate.getState().activeLabelId == null && l.length) store.setActiveLabel(l[0].id)
      })
      .catch((e) => setError(e.message))
    api.aiStatus().then(setAiStatus).catch(() => setAiStatus(null))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId])

  // ---- labeling session (effort logging) ----------------------------------
  useEffect(() => {
    let cancelled = false
    const { participant } = loadSettings(projectId)
    api.createSession(projectId, mode, participant)
      .then((session) => {
        if (cancelled) return
        effort.start(projectId, session.id)
        if (imageIdRef.current != null) effort.openImage(imageIdRef.current)
      })
      .catch((e) => setError(`Could not start session: ${e.message}`))
    return () => {
      cancelled = true
      effort.stop()
    }
  }, [projectId, mode])

  const changeMode = (next: Mode) => {
    saveSetting(projectId, 'mode', next)
    if (next === 'manual') {
      store.setProposals([])
      if (tool === 'smart') store.setTool('select')
    }
    setMode(next)
  }

  // ---- saving -------------------------------------------------------------
  const save = useCallback(async () => {
    const id = imageIdRef.current
    const { version: v, annotations: anns } = useAnnotate.getState()
    if (id == null || v === savedVersion.current) return
    setSaveState('saving')
    try {
      await api.putAnnotations(id, anns)
      savedVersion.current = v
      setSaveState('saved')
      // Mirror the server's new <-> in_progress status change.
      setImage((img) => img && img.id === id && img.status !== 'done'
        ? { ...img, status: anns.length ? 'in_progress' : 'new' } : img)
    } catch (e) {
      setSaveState('error')
      setError(`Save failed: ${(e as Error).message}`)
    }
  }, [])

  useEffect(() => {
    if (version === savedVersion.current) return
    const t = window.setTimeout(save, 600)
    return () => window.clearTimeout(t)
  }, [version, save])

  useEffect(() => () => void save(), [save])

  // ---- image loading ------------------------------------------------------
  useEffect(() => {
    let cancelled = false
    setImage(null)
    setEmbedState('idle')
    setDetecting(false)
    setPending(null)
    detectionsRef.current = []
    Promise.all([api.getImage(imageId), api.getAnnotations(imageId)])
      .then(([info, anns]) => {
        if (cancelled) return
        imageIdRef.current = imageId
        useAnnotate.getState().reset(anns)
        savedVersion.current = useAnnotate.getState().version
        setImage(info)
        effort.openImage(imageId)
      })
      .catch((e) => setError(e.message))
    return () => {
      cancelled = true
    }
  }, [imageId])

  // Precompute the SAM image embedding so the first click is fast.
  useEffect(() => {
    if (!image || !samReady) return
    let cancelled = false
    setEmbedState('running')
    api.embed(image.id)
      .then(() => !cancelled && setEmbedState('ready'))
      .catch((e) => {
        if (cancelled) return
        setEmbedState('failed')
        setError(`SAM: ${e.message}`)
      })
    return () => {
      cancelled = true
    }
  }, [image, samReady])

  // Run SAM whenever the smart-select prompt changes.
  useEffect(() => {
    if (!image || (!smartPoints.length && !smartBox)) {
      useAnnotate.getState().setSmartPreview(null)
      return
    }
    const seq = ++segmentSeq.current
    setSmartBusy(true)
    api.segment(image.id, smartPoints, smartBox)
      .then((r) => seq === segmentSeq.current && useAnnotate.getState().setSmartPreview({ polygon: r.polygon, score: r.score }))
      .catch((e) => seq === segmentSeq.current && setError(`SAM: ${e.message}`))
      .finally(() => seq === segmentSeq.current && setSmartBusy(false))
  }, [image, smartPoints, smartBox])

  // ---- creating annotations ----------------------------------------------
  /** Add a finished shape and drop detector suggestions it makes redundant. */
  const commitNew = (a: Annotation, tool: string) => {
    store.add([a])
    effort.commit([a], tool)
    if (alikeReady && imageIdRef.current != null && similar?.stage !== 'running') offerSimilar(imageIdRef.current, a)
    const box = boundingBox(a)
    for (const p of useAnnotate.getState().proposals) {
      if (boxIou(p.box, box) >= 0.5) store.removeProposal(p.id)
    }
  }

  /** The detector's name for whatever lies under this box, if it saw it. */
  const aiNameFor = (box: BoxGeom): string | null => {
    let best: string | null = null
    let bestIou = 0.5
    for (const d of detectionsRef.current) {
      const v = boxIou(box, d.box)
      if (d.score >= NAME_MIN_SCORE && v >= bestIou) {
        best = d.label
        bestIou = v
      }
    }
    return best
  }

  /**
   * Name a new shape. Simple view: the AI's name if it recognised the object, otherwise ask.
   * Expert view: the active label, otherwise ask.
   */
  const place = async (draft: Annotation, tool: string) => {
    const id = imageIdRef.current
    const guess = expert ? null : aiNameFor(boundingBox(draft))
    const labelId = guess ? await labelFor(guess) : expert ? useAnnotate.getState().activeLabelId : null
    if (imageIdRef.current !== id) return
    if (labelId == null) setPending({ draft, tool })
    else commitNew({ ...draft, label_id: labelId }, tool)
  }

  const pickName = async (choice: { id: number } | { name: string }) => {
    if (!pending) return
    const labelId = 'id' in choice ? choice.id : await labelFor(choice.name)
    if (labelId == null) return
    store.setActiveLabel(labelId)
    commitNew({ ...pending.draft, label_id: labelId }, pending.tool)
    setPending(null)
  }

  const createBox = (_kind: 'box', geometry: BoxGeom, tool: string) =>
    void place({ id: uid(), kind: 'box', geometry, label_id: -1, source: 'manual', score: null }, tool)

  const createPolygon = (geometry: PolygonGeom) => {
    store.setDraftPolygon([])
    void place({ id: uid(), kind: 'polygon', geometry, label_id: -1, source: 'manual', score: null }, 'polygon')
  }

  const acceptSmart = () => {
    const { smartPreview, smartReplaceId, annotations: anns } = useAnnotate.getState()
    if (!smartPreview || smartPreview.polygon.length < 6) return
    const replaced = anns.find((a) => a.id === smartReplaceId)
    const a: Annotation = { id: uid(), kind: 'polygon', geometry: { points: smartPreview.polygon },
                            label_id: replaced?.label_id ?? -1, source: 'ai', score: smartPreview.score }
    store.clearSmart()
    if (replaced) {
      store.replace(replaced.id, [a])
      effort.commit([a], 'refine')
    } else {
      void place(a, 'smart')
    }
  }

  /** Map a detector class name to a project label, creating the label if needed. */
  const labelFor = async (name: string): Promise<number | null> => {
    const found = labelsRef.current.find((l) => l.name.toLowerCase() === name.toLowerCase())
    if (found) return found.id
    try {
      const created = await api.createLabel(projectId, name, nextColor(labelsRef.current.map((l) => l.color)))
      setLabelList([...labelsRef.current, created])
      return created.id
    } catch (e) {
      setError((e as Error).message)
      return null
    }
  }

  /** Turn suggestions into annotations. auto = added by the assistant without a click. */
  const acceptProposals = async (items: Proposal[], auto = false) => {
    const id = imageIdRef.current
    const created: Annotation[] = []
    for (const p of items) {
      const labelId = await labelFor(p.label)
      if (imageIdRef.current !== id) return // image changed while labels were being created
      if (labelId == null) continue
      const a: Annotation = { id: uid(), kind: 'box', geometry: p.box, label_id: labelId, source: 'ai', score: p.score }
      created.push(a)
      store.removeProposal(p.id)
      effort.log('proposal_accepted', { score: p.score, label: p.label, class_id: p.class_id, auto }, a.id)
    }
    if (created.length) {
      store.add(created)
      effort.commit(created, auto ? 'auto' : 'proposal')
    }
  }

  const rejectProposals = (items: Proposal[]) => {
    for (const p of items) {
      store.removeProposal(p.id)
      effort.log('proposal_rejected', { score: p.score, label: p.label, class_id: p.class_id })
    }
  }

  /** Run YOLO on the open image. Boxes that overlap an existing annotation are not proposed again. */
  const runDetector = async () => {
    if (!image) return
    const id = image.id
    setDetecting(true)
    try {
      const res = await api.detect(id, 0.05, labelsRef.current.map((l) => l.name))
      if (imageIdRef.current !== id) return // user moved to another image meanwhile
      const existing = useAnnotate.getState().annotations.map(boundingBox)
      const fresh = res.detections
        .filter((d) => !existing.some((b) => boxIou(b, d.box) >= 0.5))
        .map((d) => ({ ...d, id: uid() }))
      detectionsRef.current = res.detections
      store.setProposals(fresh)
      // Confident boxes are added straight away; only unsure ones are left for the user to check.
      const sure = autoAccept ? fresh.filter((p) => p.score >= AUTO_ACCEPT) : []
      if (sure.length) await acceptProposals(sure, true)
      if (imageIdRef.current !== id) return
      const { confThreshold, proposals } = useAnnotate.getState()
      const review = proposals.filter((p) => p.score >= confThreshold).length
      setSuggested((s) => ({ added: sure.length, review, nonce: s.nonce + 1 }))
    } catch (e) {
      setError(`YOLO: ${(e as Error).message}`)
    } finally {
      if (imageIdRef.current === id) setDetecting(false)
    }
  }

  // Always-on assistant: suggest objects as soon as an image opens (AI mode only).
  useEffect(() => {
    if (image && yoloReady && autoSuggest && !isAiCleared(projectId, image.id)) void runDetector()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [image?.id, yoloReady, autoSuggest])

  // ---- remove AI labels ---------------------------------------------------
  const removedAi = useRef<Annotation[]>([])
  const aiLabelCount = store.annotations.filter((a) => a.source === 'ai').length

  /** Drop every untouched AI label (and pending suggestions) from the open image. */
  const removeAiLabels = (): number => {
    const ai = useAnnotate.getState().annotations.filter((a) => a.source === 'ai')
    removedAi.current = ai
    store.removeMany(ai.map((a) => a.id))
    store.setProposals([])
    // Do not let the always-on assistant label this image again on the next visit.
    if (imageIdRef.current != null) setAiCleared(projectId, imageIdRef.current, true)
    return ai.length
  }

  const undoRemoveAi = () => {
    const back = removedAi.current
    removedAi.current = []
    const present = new Set(useAnnotate.getState().annotations.map((a) => a.id))
    store.add(back.filter((a) => !present.has(a.id)))
    if (imageIdRef.current != null) setAiCleared(projectId, imageIdRef.current, false)
  }

  const changeAutoSuggest = (on: boolean) => {
    saveSetting(projectId, 'autoSuggest', on)
    setAutoSuggest(on)
  }

  const changeAutoAccept = (on: boolean) => {
    saveSetting(projectId, 'autoAccept', on)
    setAutoAccept(on)
  }

  const changeExpert = (on: boolean) => {
    saveSetting(projectId, 'expert', on)
    setExpert(on)
    if (!on && tool === 'polygon') store.setTool(samReady ? 'smart' : 'box')
  }

  // Simple view starts on Magic select: one tap on an object is the easiest way to label it.
  useEffect(() => {
    if (samReady && !expert && useAnnotate.getState().tool === 'select') store.setTool('smart')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [samReady])

  const aiActivity = mode !== 'ai' ? null
    : similar?.stage === 'running' ? `AI is looking for “${similar.name}”…`
    : detecting ? 'AI is finding objects…'
    : smartBusy ? 'AI is tracing the outline…'
    : embedState === 'running' ? 'AI is reading the image…'
    : null

  // ---- find alike objects -------------------------------------------------
  const labelName = (id: number) => labelsRef.current.find((l) => l.id === id)?.name ?? 'this'

  const offerSimilar = (imageId: number, a: Annotation) => {
    exampleRef.current = { imageId, annotation: a }
    setSimilar({ stage: 'offer', name: labelName(a.label_id) })
  }

  /**
   * Search every image of the project for objects of the same kind as the example (mark one cricket
   * ball, get all balls, also other colours and sizes; OWLv2). Results have the example's kind. The open image gets them in the
   * editor (undoable, autosaved); other images are saved by the server.
   */
  const runSimilar = async () => {
    const example = exampleRef.current
    if (!example) return
    const { imageId: sourceId, annotation: a } = example
    const region = a.kind === 'box' ? { box: a.geometry } : { polygon: a.geometry.points }
    await runSearch(labelName(a.label_id), a.label_id, a.kind, sourceId, 'similar',
                    (target, save) => api.findSimilar(target, { source_image_id: sourceId, label_id: a.label_id, save, ...region }))
  }

  /**
   * "Find by name": type "car", the AI marks every car in all images as boxes (OWLv2 text query).
   * The label is the typed name (reused when it exists).
   */
  const runTextSearch = async (text: string) => {
    if (similar?.stage === 'running') return
    const labelId = await labelFor(text)
    if (labelId == null || imageIdRef.current == null) return
    const name = labelsRef.current.find((l) => l.id === labelId)?.name ?? text
    await runSearch(name, labelId, 'box', imageIdRef.current, 'text',
                    (target, save) => api.textSearch(target, { text, label_id: labelId, save }))
  }

  /** Run a search on every image (the start image first). Shared by find alike and find by name. */
  const runSearch = async (name: string, labelId: number, kind: 'box' | 'polygon', startId: number,
                           tool: 'similar' | 'text', call: (target: number, save: boolean) => Promise<SimilarResult>) => {
    const order = [startId, ...images.map((i) => i.id).filter((id) => id !== startId)]
    similarStop.current = false
    similarIds.current = []
    let found = 0
    let touched = 0
    setSimilar({ stage: 'running', name, done: 0, total: order.length, found })
    for (let i = 0; i < order.length && !similarStop.current; i++) {
      const target = order[i]
      try {
        const local = target === imageIdRef.current
        let regions = (await call(target, !local)).regions
        if (local && imageIdRef.current !== target) {
          // The user moved on while this image was searched: let the server save the result.
          regions = (await call(target, true)).regions
        } else if (local && regions.length) {
          const created: Annotation[] = regions.map((r) => kind === 'box'
            ? { id: uid(), kind: 'box', geometry: r.box, label_id: labelId, source: 'ai', score: r.score }
            : { id: uid(), kind: 'polygon', geometry: { points: r.polygon }, label_id: labelId, source: 'ai', score: r.score })
          store.add(created)
          effort.commit(created, tool)
          regions = created.map((c, k) => ({ ...regions[k], id: c.id }))
        }
        similarIds.current.push(...regions.map((r) => r.id!).filter(Boolean))
        found += regions.length
        if (regions.length) touched++
      } catch (e) {
        setError(`${tool === 'text' ? 'Find by name' : 'Find alike'}: ${(e as Error).message}`)
        break
      }
      setSimilar({ stage: 'running', name, done: i + 1, total: order.length, found })
    }
    setSimilar({ stage: 'result', name, found, images: touched, stopped: similarStop.current })
    if (touched) api.listImages(projectId).then(setImages).catch(() => {})
  }

  const undoSimilar = async () => {
    const ids = similarIds.current
    similarIds.current = []
    const open = new Set(useAnnotate.getState().annotations.map((x) => x.id))
    for (const id of ids) if (open.has(id)) store.remove(id)
    try {
      await save()
      await api.undoSimilar(ids)
      api.listImages(projectId).then(setImages).catch(() => {})
    } catch (e) {
      setError(`Undo: ${(e as Error).message}`)
    }
    setSimilar(null)
  }

  // ---- navigation ---------------------------------------------------------
  const index = images.findIndex((i) => i.id === imageId)
  const goTo = async (id: number | undefined) => {
    if (id == null) return
    await save()
    navigate(`/projects/${projectId}/annotate/${id}`)
  }

  const markDone = async () => {
    await save()
    try {
      await api.setImageStatus(imageId, 'done')
      effort.log('image_done', { annotations: useAnnotate.getState().annotations.length })
      const list = await api.listImages(projectId)
      setImages(list)
      const pos = list.findIndex((i) => i.id === imageId)
      const next = [...list.slice(pos + 1), ...list.slice(0, pos)].find((i) => i.status !== 'done')
      if (next) navigate(`/projects/${projectId}/annotate/${next.id}`)
      else navigate(`/projects/${projectId}`)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  /** Save first so the download includes the latest edits, then fetch the rendered JPEG. */
  const downloadAnnotated = async () => {
    await save()
    const a = document.createElement('a')
    a.href = api.renderedUrl(imageId)
    a.download = ''
    document.body.appendChild(a)
    a.click()
    a.remove()
  }

  // ---- keyboard -----------------------------------------------------------
  const keyHandler = useRef<(e: KeyboardEvent) => void>(() => {})
  keyHandler.current = (e: KeyboardEvent) => {
    if ((e.target as HTMLElement).closest('input, textarea, select')) return
    const s = useAnnotate.getState()
    const ctrl = e.ctrlKey || e.metaKey
    const tools: Record<string, Tool> = { v: 'select', b: 'box', p: 'polygon', h: 'pan', s: 'smart' }
    const key = e.key.toLowerCase()

    if (ctrl && key === 'z') { e.preventDefault(); if (e.shiftKey) s.redo(); else s.undo(); return }
    if (ctrl && key === 'y') { e.preventDefault(); s.redo(); return }
    if (ctrl) return

    if (tools[key]) {
      if (key === 's' && !samReady) return
      if (key === 'p' && !expert) return
      s.setTool(tools[key])
    } else if (e.key === 'Enter') {
      if (s.tool === 'polygon' && s.draftPolygon.length >= 6) createPolygon({ points: s.draftPolygon })
      else if (s.tool === 'smart') acceptSmart()
    } else if (e.key === 'Escape') {
      s.setDraftPolygon([])
      s.clearSmart()
      s.select(null)
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault()
      if (s.tool === 'polygon' && s.draftPolygon.length) s.setDraftPolygon(s.draftPolygon.slice(0, -2))
      else if (s.tool === 'smart' && s.smartPoints.length) s.setSmartPrompt(s.smartPoints.slice(0, -1), s.smartBox)
      else if (s.selectedProposalId) rejectProposals(s.proposals.filter((p) => p.id === s.selectedProposalId))
      else if (s.selectedId) s.remove(s.selectedId)
    } else if (key === 'a' && s.selectedProposalId) {
      acceptProposals(s.proposals.filter((p) => p.id === s.selectedProposalId))
    } else if (/^[1-9]$/.test(e.key)) {
      const label = labelsRef.current[Number(e.key) - 1]
      if (!label) return
      s.setActiveLabel(label.id)
      if (s.selectedId) s.update(s.selectedId, { label_id: label.id })
    } else if (e.key === '?') {
      tour.start()
    } else if (e.key === '.') {
      goTo(images[index + 1]?.id)
    } else if (e.key === ',') {
      goTo(images[index - 1]?.id)
    }
  }
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => keyHandler.current(e)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // ---- render -------------------------------------------------------------
  return (
    <div className="flex h-dvh flex-col bg-neutral-950">
      <header className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-neutral-800 px-3 py-2 text-sm">
        <Link to={`/projects/${projectId}`} className="text-neutral-400 hover:text-neutral-100">← Project</Link>
        <span className="min-w-0 max-w-[40vw] truncate font-medium md:max-w-none" title={image?.filename}>{image?.filename ?? '…'}</span>
        {image && (
          <span className={`rounded px-1.5 py-0.5 text-xs ${statusStyle[image.status]}`}>{image.status.replace('_', ' ')}</span>
        )}
        <span className="text-neutral-500">{index >= 0 ? `${index + 1} / ${images.length}` : ''}</span>
        <div data-tour="nav" className="flex gap-1">
          <button className={btnSecondary} onClick={() => goTo(images[index - 1]?.id)} disabled={index <= 0}>‹ Prev</button>
          <button className={btnSecondary} onClick={() => goTo(images[index + 1]?.id)}
                  disabled={index < 0 || index >= images.length - 1}>Next ›</button>
        </div>

        <div className="ml-auto flex flex-wrap items-center gap-2 md:gap-3">
          {expert && <div data-tour="mode" className="flex rounded-lg border border-neutral-800 bg-neutral-900 p-0.5 text-xs">
            {(['ai', 'manual'] as Mode[]).map((m) => (
              <button key={m} onClick={() => changeMode(m)}
                      className={`rounded-md px-2 py-1 transition duration-200 ${mode === m ? 'bg-neutral-100 text-neutral-950 shadow-sm' : 'text-neutral-400 hover:text-neutral-100'}`}>
                {m === 'ai' ? 'AI-assisted' : 'Manual'}
              </button>
            ))}
          </div>}
          <span data-tour="save" className={`text-xs ${saveState === 'error' ? 'text-red-400' : 'text-neutral-500'}`}>
            {saveState === 'saving' ? 'Saving…' : saveState === 'error' ? 'Save failed' : 'Saved ✓'}
          </span>
          <label data-tour="expert" title="Show every tool and setting"
                 className="flex cursor-pointer select-none items-center gap-1.5 text-xs text-neutral-400 hover:text-neutral-100">
            <input type="checkbox" checked={expert} onChange={(e) => changeExpert(e.target.checked)}
                   className="accent-white" />
            Expert tools
          </label>
          {expert && (
            <button data-tour="download" className={btnSecondary} onClick={downloadAnnotated} disabled={!image}
                    title="Download this image with its annotations drawn on">⤓ Image</button>
          )}
          <button data-tour="done" className={btnPrimary} onClick={markDone} disabled={!image}>
            {expert ? 'Mark done ✓' : 'Done, next image ›'}
          </button>
          <HelpButton onClick={tour.start} />
        </div>
      </header>

      {error && (
        <div className="flex items-center justify-between bg-red-500/10 px-3 py-1.5 text-sm text-red-300">
          <span>{error}</span>
          <button className="text-red-300 hover:text-white" onClick={() => setError(null)}>✕</button>
        </div>
      )}

      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <Toolbar tool={tool} onTool={store.setTool} smartEnabled={samReady} simple={!expert}
                 onUndo={store.undo} onRedo={store.redo}
                 canUndo={store.past.length > 0} canRedo={store.future.length > 0} />

        <div data-tour="canvas" className="relative min-h-[45dvh] min-w-0 flex-1 md:min-h-0">
          {image ? (
            <AnnotationCanvas
              imageUrl={api.imageUrl(image.id)}
              imageWidth={image.width}
              imageHeight={image.height}
              labels={labels}
              onCreate={createBox}
              onCreatePolygon={createPolygon}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-neutral-500">Loading…</div>
          )}
          <AiWorking label={aiActivity} />
          <AiSuggestionsToast added={suggested.added} review={suggested.review} nonce={suggested.nonce} />
          {similar && mode === 'ai' && (
            <SimilarSearch state={similar} onFind={() => void runSimilar()}
                           onStop={() => { similarStop.current = true }}
                           onUndo={() => void undoSimilar()} onClose={() => setSimilar(null)} />
          )}
          {pending && (
            <NamePicker labels={labels} lastId={activeLabelId}
                        onPick={(c) => void pickName(c)} onCancel={() => setPending(null)} />
          )}
          <RemoveAiButton key={image?.id} count={aiLabelCount} onRemove={removeAiLabels} onUndo={undoRemoveAi} />
          <ToolToast tool={tool} />
          <ToolHint tool={tool} smartBusy={smartBusy} embedState={embedState}
                    hasPreview={!!store.smartPreview} score={store.smartPreview?.score} onAccept={acceptSmart} onClear={store.clearSmart} />
        </div>

        <aside className="max-h-[40dvh] w-full shrink-0 space-y-4 overflow-y-auto border-t border-neutral-800 p-3 md:max-h-none md:w-80 md:border-l md:border-t-0">
          {expert && <section data-tour="labels" className="space-y-1.5">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-400">Labels</h3>
            {labels.length === 0 && (
              <p className="text-xs text-neutral-500">No labels. Add some on the project page.</p>
            )}
            <div className="flex flex-wrap gap-1.5">
              {labels.map((l, i) => (
                <button key={l.id} onClick={() => store.setActiveLabel(l.id)}
                        className={`flex items-center gap-1.5 rounded border px-2 py-1 text-xs ${
                          activeLabelId === l.id ? 'border-neutral-200 bg-neutral-800' : 'border-neutral-700'}`}>
                  <span className="h-2.5 w-2.5 rounded-sm" style={{ background: l.color }} />
                  {l.name}
                  {i < 9 && <span className="text-neutral-500">{i + 1}</span>}
                </button>
              ))}
            </div>
          </section>}

          {textReady && (
            <TextSearch busy={similar?.stage === 'running'} onSearch={(t) => void runTextSearch(t)} />
          )}

          {mode === 'ai' ? (
            <div data-tour="autolabel">
            <ProposalPanel
              available={yoloReady}
              detail={aiStatus?.yolo.detail}
              detecting={detecting}
              simple={!expert}
              autoSuggest={autoSuggest}
              onAutoSuggest={changeAutoSuggest}
              autoAccept={autoAccept}
              onAutoAccept={changeAutoAccept}
              onDetect={runDetector}
              onAccept={acceptProposals}
              onReject={rejectProposals}
            />
            </div>
          ) : (
            <p className="rounded-md border border-neutral-800 p-2 text-xs text-neutral-400">
              Manual mode: AI tools are off. Use this mode for the baseline of the effort study.
            </p>
          )}

          <div data-tour="annotations">
            <AnnotationList labels={labels} samEnabled={samReady} simple={!expert} alikeEnabled={alikeReady}
                            onFindSimilar={(a) => imageIdRef.current != null && offerSimilar(imageIdRef.current, a)} />
          </div>

          {expert && <details className="text-xs text-neutral-400">
            <summary className="cursor-pointer text-neutral-300">Shortcuts</summary>
            <ul className="mt-2 space-y-0.5">
              <li>V select · B box · P polygon · S smart (SAM) · H pan · F fit</li>
              <li>Polygon: click points, Enter or click first point to close</li>
              <li>Smart: click = add object point, right/Alt-click = background point, drag = box, Enter = accept</li>
              <li>Del delete / remove last point · Esc cancel</li>
              <li>A accept selected proposal · 1–9 pick label</li>
              <li>Ctrl+Z undo · Ctrl+Y redo · , / . previous / next image</li>
              <li>Alt+click a polygon vertex to remove it</li>
              <li>? show the tutorial</li>
            </ul>
          </details>}
        </aside>
      </div>

      <Tour steps={ANNOTATE_TOUR} open={tour.open} onClose={tour.close} />
    </div>
  )
}

/** Short animated card naming the tool after every tool switch. */
function ToolToast({ tool }: { tool: Tool }) {
  const prev = useRef(tool)
  const [shown, setShown] = useState<{ tool: Tool; n: number } | null>(null)
  useEffect(() => {
    if (prev.current === tool) return
    prev.current = tool
    setShown((s) => ({ tool, n: (s?.n ?? 0) + 1 }))
  }, [tool])
  const t = shown && TOOLS.find((x) => x.id === shown.tool)
  if (!shown || !t) return null
  return (
    <div key={shown.n} role="status"
         className="pointer-events-none absolute bottom-16 left-1/2 z-10 flex max-w-[90%] animate-toast items-center gap-3 rounded-xl border border-neutral-700 bg-neutral-900/95 px-3 py-2 shadow-xl shadow-black/50 backdrop-blur">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white text-base text-neutral-950">{t.icon}</span>
      <span className="min-w-0">
        <span className="block font-display text-sm font-semibold text-white">
          {t.label} <kbd className="ml-1 font-mono text-xs text-neutral-500">{t.key}</kbd>
        </span>
        <span className="line-clamp-2 block text-xs text-neutral-400">{t.tip}</span>
      </span>
    </div>
  )
}

const ANNOTATE_TOUR: TourStep[] = [
  { target: 'canvas', title: 'Your image', body: 'Everything you mark shows up right on the picture, with its name in a coloured tag. The AI starts looking for objects as soon as the image opens.' },
  ...TOOLS.map((t) => ({ target: `tool-${t.id}`, title: `${t.icon}  ${t.label}`, body: `${t.tip} Shortcut: ${t.key}.` })),
  { target: 'history', title: 'Undo / redo', body: 'Step back or forward through your edits. Ctrl+Z and Ctrl+Y work too.' },
  { target: 'zoom', title: 'Zoom', body: 'Zoom in and out, or press Fit (F) to see the whole image again.' },
  { target: 'labels', title: 'Labels', body: 'Pick the label for new shapes. Keys 1-9 choose a label and relabel the selected shape.' },
  { target: 'autolabel', title: 'Your AI assistant', body: 'The AI checks every image you open. Objects it is sure about are marked for you. When it is unsure it asks: answer Yes or No. A glowing border means it is working.' },
  { target: 'annotations', title: 'Objects on this image', body: 'Everything marked so far. Fix a name with the menu, remove a mark with ✕, or tap ⧉ so the AI marks every alike object in all images (mark one ball, it finds the others).' },
  { target: 'expert', title: 'Expert tools', body: 'Need more control? Turn this on for extra drawing tools, label setup, the confidence slider and keyboard shortcuts.' },
  { target: 'mode', title: 'AI-assisted vs manual', body: 'Manual turns the AI tools off. Time and clicks are logged per mode to measure the effort AI saves.' },
  { target: 'remove-ai', title: 'Remove AI labels', body: 'Did the AI label things you do not want? This removes everything it added to this image in one go. Labels you made or fixed stay, and Undo brings them back.' },
  { target: 'text-search', title: 'Find by name', body: 'Type what you are looking for, like “car” or “helmet”. The AI marks every one in all images, also things the auto-label model does not know.' },
  { target: 'save', title: 'Autosave', body: 'Every change saves by itself. There is no save button to forget.' },
  { target: 'download', title: 'Download annotated image', body: 'Saves this image as a JPEG with all its annotations drawn on.' },
  { target: 'done', title: 'Done', body: 'When everything is marked, press this. Your work is saved and the next unfinished image opens.' },
  { target: 'nav', title: 'Previous / next', body: 'Move between images. The , and . keys do the same.' },
  { target: 'help', title: 'Replay this tour', body: 'Tap ? (or press ?) any time to see this tutorial again.' },
]

function ToolHint({ tool, smartBusy, embedState, hasPreview, score, onAccept, onClear }: {
  tool: Tool
  smartBusy: boolean
  embedState: string
  hasPreview: boolean
  /** SAM's own estimate of the outline's IoU (0-1). */
  score?: number
  onAccept: () => void
  onClear: () => void
}) {
  if (tool !== 'smart') return null
  return (
    <div className="absolute left-1/2 top-3 flex -translate-x-1/2 items-center gap-2 rounded-md bg-neutral-800/90 px-3 py-1.5 text-xs text-neutral-200 shadow">
      {embedState === 'running' ? 'Getting the image ready…'
        : smartBusy ? 'Outlining…'
        : hasPreview ? 'Outline wrong? Tap the part that is missing.'
        : 'Tap any object. AI outlines it for you.'}
      {hasPreview && score != null && <ConfidenceChip score={score} />}
      {hasPreview && (
        <>
          <button className="rounded bg-cyan-600 px-2 py-0.5 text-white hover:bg-cyan-500" onClick={onAccept}>✓ Looks right</button>
          <button className="rounded px-2 py-0.5 hover:bg-neutral-700" onClick={onClear}>Start over</button>
        </>
      )}
    </div>
  )
}

/** How sure SAM is about its outline (its predicted IoU), in plain words. */
function ConfidenceChip({ score }: { score: number }) {
  const pctSure = Math.round(Math.min(Math.max(score, 0), 1) * 100)
  const [style, word] = score >= 0.85 ? ['bg-emerald-500/15 text-emerald-300', 'sure']
    : score >= 0.6 ? ['bg-amber-500/15 text-amber-300', 'check the edges']
    : ['bg-red-500/15 text-red-300', 'unsure, tap the missing part']
  return (
    <span className={`rounded px-1.5 py-0.5 font-medium ${style}`}
          title="The AI's own estimate of how well the outline fits (predicted IoU)">
      ✦ {pctSure}% {word}
    </span>
  )
}
