import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'
import { api } from '../api/client'
import type { AiStatus, Annotation, BoxGeom, ImageInfo, Label, Mode, PolygonGeom } from '../api/types'
import { AnnotationCanvas } from '../components/AnnotationCanvas'
import { AnnotationList } from '../components/AnnotationList'
import { ProposalPanel } from '../components/ProposalPanel'
import { Toolbar } from '../components/Toolbar'
import { effort } from '../lib/effort'
import { nextColor, uid } from '../lib/geometry'
import { loadSettings, saveSettings } from '../lib/settings'
import { btnPrimary, btnSecondary, statusStyle } from '../lib/ui'
import { useAnnotate, type Proposal, type Tool } from '../store/annotate'

type SaveState = 'saved' | 'saving' | 'error'

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

  const store = useAnnotate()
  const { version, tool, activeLabelId, smartPoints, smartBox } = store
  const imageIdRef = useRef<number | null>(null)
  const savedVersion = useRef(0)
  const segmentSeq = useRef(0)

  const samReady = mode === 'ai' && !!aiStatus?.sam.available
  const yoloReady = mode === 'ai' && !!aiStatus?.yolo.available

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
    saveSettings(projectId, { ...loadSettings(projectId), mode: next })
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
  const requireLabel = (): number | null => {
    const id = useAnnotate.getState().activeLabelId
    if (id == null) setError('Add a label on the project page, then pick it in the side panel.')
    return id
  }

  const createBox = (_kind: 'box', geometry: BoxGeom, tool: string) => {
    const labelId = requireLabel()
    if (labelId == null) return
    const a: Annotation = { id: uid(), kind: 'box', geometry, label_id: labelId, source: 'manual', score: null }
    store.add([a])
    effort.commit([a], tool)
  }

  const createPolygon = (geometry: PolygonGeom) => {
    const labelId = requireLabel()
    if (labelId == null) return
    const a: Annotation = { id: uid(), kind: 'polygon', geometry, label_id: labelId, source: 'manual', score: null }
    store.setDraftPolygon([])
    store.add([a])
    effort.commit([a], 'polygon')
  }

  const acceptSmart = () => {
    const { smartPreview, smartReplaceId, annotations: anns } = useAnnotate.getState()
    if (!smartPreview || smartPreview.polygon.length < 6) return
    const replaced = anns.find((a) => a.id === smartReplaceId)
    const labelId = replaced?.label_id ?? requireLabel()
    if (labelId == null) return
    const a: Annotation = { id: uid(), kind: 'polygon', geometry: { points: smartPreview.polygon },
                            label_id: labelId, source: 'ai', score: smartPreview.score }
    store.replace(replaced?.id ?? null, [a])
    store.clearSmart()
    effort.commit([a], replaced ? 'refine' : 'smart')
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

  const acceptProposals = async (items: Proposal[]) => {
    const created: Annotation[] = []
    for (const p of items) {
      const labelId = await labelFor(p.label)
      if (labelId == null) continue
      const a: Annotation = { id: uid(), kind: 'box', geometry: p.box, label_id: labelId, source: 'ai', score: p.score }
      created.push(a)
      store.removeProposal(p.id)
      effort.log('proposal_accepted', { score: p.score, label: p.label, class_id: p.class_id }, a.id)
    }
    if (created.length) {
      store.add(created)
      effort.commit(created, 'proposal')
    }
  }

  const rejectProposals = (items: Proposal[]) => {
    for (const p of items) {
      store.removeProposal(p.id)
      effort.log('proposal_rejected', { score: p.score, label: p.label, class_id: p.class_id })
    }
  }

  const runDetector = async () => {
    if (!image) return
    setDetecting(true)
    try {
      const res = await api.detect(image.id, 0.05)
      store.setProposals(res.detections.map((d) => ({ ...d, id: uid() })))
    } catch (e) {
      setError(`YOLO: ${(e as Error).message}`)
    } finally {
      setDetecting(false)
    }
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
    <div className="flex h-screen flex-col bg-slate-950">
      <header className="flex items-center gap-3 border-b border-slate-800 px-3 py-2 text-sm">
        <Link to={`/projects/${projectId}`} className="text-slate-400 hover:text-slate-100">← Project</Link>
        <span className="truncate font-medium" title={image?.filename}>{image?.filename ?? '…'}</span>
        {image && (
          <span className={`rounded px-1.5 py-0.5 text-xs ${statusStyle[image.status]}`}>{image.status.replace('_', ' ')}</span>
        )}
        <span className="text-slate-500">{index >= 0 ? `${index + 1} / ${images.length}` : ''}</span>
        <div className="flex gap-1">
          <button className={btnSecondary} onClick={() => goTo(images[index - 1]?.id)} disabled={index <= 0}>‹ Prev</button>
          <button className={btnSecondary} onClick={() => goTo(images[index + 1]?.id)}
                  disabled={index < 0 || index >= images.length - 1}>Next ›</button>
        </div>

        <div className="ml-auto flex items-center gap-3">
          <div className="flex rounded-md border border-slate-700 p-0.5 text-xs">
            {(['ai', 'manual'] as Mode[]).map((m) => (
              <button key={m} onClick={() => changeMode(m)}
                      className={`rounded px-2 py-1 ${mode === m ? 'bg-indigo-600 text-white' : 'text-slate-300'}`}>
                {m === 'ai' ? 'AI-assisted' : 'Manual'}
              </button>
            ))}
          </div>
          <span className={`text-xs ${saveState === 'error' ? 'text-red-400' : 'text-slate-500'}`}>
            {saveState === 'saving' ? 'Saving…' : saveState === 'error' ? 'Save failed' : 'Saved'}
          </span>
          <button className={btnPrimary} onClick={markDone} disabled={!image}>Mark done ✓</button>
        </div>
      </header>

      {error && (
        <div className="flex items-center justify-between bg-red-500/10 px-3 py-1.5 text-sm text-red-300">
          <span>{error}</span>
          <button className="text-red-300 hover:text-white" onClick={() => setError(null)}>✕</button>
        </div>
      )}

      <div className="flex min-h-0 flex-1">
        <Toolbar tool={tool} onTool={store.setTool} smartEnabled={samReady}
                 onUndo={store.undo} onRedo={store.redo}
                 canUndo={store.past.length > 0} canRedo={store.future.length > 0} />

        <div className="relative min-w-0 flex-1">
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
            <div className="flex h-full items-center justify-center text-slate-500">Loading…</div>
          )}
          <ToolHint tool={tool} smartBusy={smartBusy} embedState={embedState}
                    hasPreview={!!store.smartPreview} onAccept={acceptSmart} onClear={store.clearSmart} />
        </div>

        <aside className="w-80 shrink-0 space-y-4 overflow-y-auto border-l border-slate-800 p-3">
          <section className="space-y-1.5">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Labels</h3>
            {labels.length === 0 && (
              <p className="text-xs text-slate-500">No labels. Add some on the project page.</p>
            )}
            <div className="flex flex-wrap gap-1.5">
              {labels.map((l, i) => (
                <button key={l.id} onClick={() => store.setActiveLabel(l.id)}
                        className={`flex items-center gap-1.5 rounded border px-2 py-1 text-xs ${
                          activeLabelId === l.id ? 'border-slate-200 bg-slate-800' : 'border-slate-700'}`}>
                  <span className="h-2.5 w-2.5 rounded-sm" style={{ background: l.color }} />
                  {l.name}
                  {i < 9 && <span className="text-slate-500">{i + 1}</span>}
                </button>
              ))}
            </div>
          </section>

          {mode === 'ai' ? (
            <ProposalPanel
              available={yoloReady}
              detail={aiStatus?.yolo.detail}
              detecting={detecting}
              onDetect={runDetector}
              onAccept={acceptProposals}
              onReject={rejectProposals}
            />
          ) : (
            <p className="rounded-md border border-slate-800 p-2 text-xs text-slate-400">
              Manual mode: AI tools are off. Use this mode for the baseline of the effort study.
            </p>
          )}

          <AnnotationList labels={labels} samEnabled={samReady} />

          <details className="text-xs text-slate-400">
            <summary className="cursor-pointer text-slate-300">Shortcuts</summary>
            <ul className="mt-2 space-y-0.5">
              <li>V select · B box · P polygon · S smart (SAM) · H pan · F fit</li>
              <li>Polygon: click points, Enter or click first point to close</li>
              <li>Smart: click = add object point, right/Alt-click = background point, drag = box, Enter = accept</li>
              <li>Del delete / remove last point · Esc cancel</li>
              <li>A accept selected proposal · 1–9 pick label</li>
              <li>Ctrl+Z undo · Ctrl+Y redo · , / . previous / next image</li>
              <li>Alt+click a polygon vertex to remove it</li>
            </ul>
          </details>
        </aside>
      </div>
    </div>
  )
}

function ToolHint({ tool, smartBusy, embedState, hasPreview, onAccept, onClear }: {
  tool: Tool
  smartBusy: boolean
  embedState: string
  hasPreview: boolean
  onAccept: () => void
  onClear: () => void
}) {
  if (tool !== 'smart') return null
  return (
    <div className="absolute left-1/2 top-3 flex -translate-x-1/2 items-center gap-2 rounded-md bg-slate-800/90 px-3 py-1.5 text-xs text-slate-200 shadow">
      {embedState === 'running' ? 'Preparing image for SAM…'
        : smartBusy ? 'Segmenting…'
        : hasPreview ? 'Add points to refine, Enter to accept'
        : 'Click an object or drag a box'}
      {hasPreview && (
        <>
          <button className="rounded bg-cyan-600 px-2 py-0.5 text-white hover:bg-cyan-500" onClick={onAccept}>Accept</button>
          <button className="rounded px-2 py-0.5 hover:bg-slate-700" onClick={onClear}>Clear</button>
        </>
      )}
    </div>
  )
}
