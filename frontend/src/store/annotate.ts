import { create } from 'zustand'
import type { Annotation, BoxGeom, Detection, PromptPoint } from '../api/types'
import { effort } from '../lib/effort'

export type Tool = 'select' | 'box' | 'polygon' | 'smart' | 'pan'

export interface Proposal extends Detection {
  id: string
}

interface AnnotateState {
  annotations: Annotation[]
  past: Annotation[][]
  future: Annotation[][]
  /** Bumped on every user change; the page autosaves when it moves. */
  version: number
  selectedId: string | null
  tool: Tool
  activeLabelId: number | null

  draftPolygon: number[]

  smartPoints: PromptPoint[]
  smartBox: BoxGeom | null
  smartPreview: { polygon: number[]; score: number } | null
  /** Annotation replaced when the smart preview is accepted ("refine box with SAM"). */
  smartReplaceId: string | null

  proposals: Proposal[]
  selectedProposalId: string | null
  confThreshold: number

  reset: (annotations: Annotation[]) => void
  setTool: (tool: Tool) => void
  setActiveLabel: (id: number | null) => void
  select: (id: string | null) => void
  add: (items: Annotation[]) => void
  /** Remove one annotation (optional) and add new ones as a single undo step. */
  replace: (removeId: string | null, items: Annotation[]) => void
  update: (id: string, change: Partial<Annotation>) => void
  /** Change an object's attributes. Not a fix of the AI's shape or name, so the source stays. */
  setAttributes: (id: string, attributes: Annotation['attributes']) => void
  remove: (id: string) => void
  /** Remove several annotations as one undo step. */
  removeMany: (ids: string[]) => void
  undo: () => void
  redo: () => void
  setDraftPolygon: (points: number[]) => void
  setSmartPrompt: (points: PromptPoint[], box: BoxGeom | null) => void
  setSmartPreview: (preview: { polygon: number[]; score: number } | null) => void
  startRefine: (annotationId: string, box: BoxGeom) => void
  clearSmart: () => void
  setProposals: (proposals: Proposal[]) => void
  removeProposal: (id: string) => void
  selectProposal: (id: string | null) => void
  setConfThreshold: (v: number) => void
}

const HISTORY_LIMIT = 100

const smartCleared = { smartPoints: [], smartBox: null, smartPreview: null, smartReplaceId: null }

export const useAnnotate = create<AnnotateState>((set, get) => {
  /** Replace annotations and record the previous list for undo. */
  const commit = (next: Annotation[]) =>
    set((s) => ({
      annotations: next,
      past: [...s.past, s.annotations].slice(-HISTORY_LIMIT),
      future: [],
      version: s.version + 1,
    }))

  return {
    annotations: [],
    past: [],
    future: [],
    version: 0,
    selectedId: null,
    tool: 'select',
    activeLabelId: null,
    draftPolygon: [],
    ...smartCleared,
    proposals: [],
    selectedProposalId: null,
    confThreshold: 0.35,

    reset: (annotations) =>
      set({ annotations, past: [], future: [], selectedId: null, draftPolygon: [], ...smartCleared,
            proposals: [], selectedProposalId: null }),
    setTool: (tool) => set({ tool, draftPolygon: [], ...smartCleared }),
    setActiveLabel: (activeLabelId) => set({ activeLabelId }),
    select: (selectedId) => set({ selectedId, selectedProposalId: null }),

    add: (items) => commit([...get().annotations, ...items]),
    replace: (removeId, items) =>
      commit([...get().annotations.filter((a) => a.id !== removeId), ...items]),

    update: (id, change) => {
      const before = get().annotations.find((a) => a.id === id)
      if (!before) return
      const source = before.source === 'ai' ? 'ai_edited' : before.source
      const after = { ...before, ...change, source } as Annotation
      commit(get().annotations.map((a) => (a.id === id ? after : a)))
      effort.log('annotation_edited', { source_before: before.source, kind: before.kind,
                                        what: Object.keys(change) }, id)
    },

    setAttributes: (id, attributes) =>
      commit(get().annotations.map((a) => (a.id === id ? { ...a, attributes } as Annotation : a))),

    removeMany: (ids) => {
      const drop = new Set(ids)
      const removed = get().annotations.filter((a) => drop.has(a.id))
      if (!removed.length) return
      commit(get().annotations.filter((a) => !drop.has(a.id)))
      set({ selectedId: drop.has(get().selectedId ?? '') ? null : get().selectedId })
      for (const a of removed) effort.log('annotation_deleted', { source: a.source, kind: a.kind, bulk: true }, a.id)
    },

    remove: (id) => {
      const target = get().annotations.find((a) => a.id === id)
      if (!target) return
      commit(get().annotations.filter((a) => a.id !== id))
      set({ selectedId: get().selectedId === id ? null : get().selectedId })
      effort.log('annotation_deleted', { source: target.source, kind: target.kind }, id)
    },

    undo: () => {
      const { past, annotations, future, version } = get()
      if (!past.length) return
      set({ annotations: past[past.length - 1], past: past.slice(0, -1), future: [annotations, ...future],
            version: version + 1, selectedId: null })
    },
    redo: () => {
      const { past, annotations, future, version } = get()
      if (!future.length) return
      set({ annotations: future[0], future: future.slice(1), past: [...past, annotations],
            version: version + 1, selectedId: null })
    },

    setDraftPolygon: (draftPolygon) => set({ draftPolygon }),
    setSmartPrompt: (smartPoints, smartBox) => set({ smartPoints, smartBox }),
    setSmartPreview: (smartPreview) => set({ smartPreview }),
    startRefine: (annotationId, box) =>
      set({ tool: 'smart', draftPolygon: [], smartPoints: [], smartBox: box, smartPreview: null,
            smartReplaceId: annotationId }),
    clearSmart: () => set(smartCleared),

    setProposals: (proposals) => set({ proposals, selectedProposalId: null }),
    removeProposal: (id) =>
      set((s) => ({ proposals: s.proposals.filter((p) => p.id !== id),
                    selectedProposalId: s.selectedProposalId === id ? null : s.selectedProposalId })),
    selectProposal: (selectedProposalId) => set({ selectedProposalId, selectedId: null }),
    setConfThreshold: (confThreshold) => set({ confThreshold }),
  }
})
