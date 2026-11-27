import type { Annotation, Attributes, Label, Source } from '../api/types'
import { useAnnotate } from '../store/annotate'

const SOURCE_BADGE: Record<Source, { text: string; cls: string }> = {
  manual: { text: 'you', cls: 'bg-neutral-700 text-neutral-200' },
  ai: { text: '✦ AI', cls: 'bg-cyan-500/20 text-cyan-300' },
  ai_edited: { text: '✦ AI · fixed', cls: 'bg-amber-500/20 text-amber-300' },
}

export function AnnotationList({ labels, samEnabled, alikeEnabled, simple, onFindSimilar }: {
  labels: Label[]
  samEnabled: boolean
  alikeEnabled: boolean
  simple: boolean
  onFindSimilar: (a: Annotation) => void
}) {
  const { annotations, selectedId, select, update, remove, startRefine, setAttributes } = useAnnotate()
  const colorOf = (id: number) => labels.find((l) => l.id === id)?.color ?? '#94a3b8'

  return (
    <section className="space-y-1.5">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
        {simple ? 'Objects on this image' : 'Annotations'} ({annotations.length})
      </h3>
      {annotations.length === 0 && (
        <p className="text-xs text-neutral-500">
          {simple ? 'Nothing marked yet. Tap an object with ✦ Magic select, or wait for the AI.' : 'Nothing labeled on this image yet.'}
        </p>
      )}
      <ul className="space-y-1">
        {annotations.map((a) => {
          const badge = SOURCE_BADGE[a.source]
          const at = a.attributes ?? {}
          return (
            <li key={a.id} className="space-y-1">
            <div onClick={() => { select(a.id); useAnnotate.getState().setTool('select') }}
                className={`flex cursor-pointer items-center gap-1.5 rounded px-1.5 py-1 text-xs ${
                  a.id === selectedId ? 'bg-neutral-800 ring-1 ring-neutral-600' : 'hover:bg-neutral-900'}`}>
              <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: colorOf(a.label_id) }} />
              <select value={a.label_id} onClick={(e) => e.stopPropagation()}
                      onChange={(e) => update(a.id, { label_id: Number(e.target.value) })}
                      className="min-w-0 flex-1 rounded border border-neutral-700 bg-neutral-900 px-1 py-0.5">
                {labels.map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
              </select>
              {!simple && <span className="text-neutral-500">{a.kind === 'box' ? '▭' : '⬠'}</span>}
              <span className={`rounded px-1 py-0.5 ${badge.cls}`}>{badge.text}</span>
              {a.kind === 'box' && samEnabled && (
                <button title="Outline this object exactly with AI" className="text-cyan-300 hover:text-cyan-200"
                        onClick={(e) => { e.stopPropagation(); startRefine(a.id, a.geometry) }}>✦</button>
              )}
              {alikeEnabled && (
                <button title="Find every object of this kind in all images" aria-label="Find alike objects"
                        className="text-cyan-300 hover:text-cyan-200"
                        onClick={(e) => { e.stopPropagation(); onFindSimilar(a) }}>⧉</button>
              )}
              {at.truncated && <span title="Cut off at the image edge" className="text-neutral-400">⇥</span>}
              {at.occluded && <span title="Partly hidden" className="text-neutral-400">◐</span>}
              <button title="Delete" className="text-neutral-500 hover:text-red-400"
                      onClick={(e) => { e.stopPropagation(); remove(a.id) }}>✕</button>
            </div>
            {a.id === selectedId && (
              <AttributeEditor value={at} onChange={(next) => setAttributes(a.id, next)} />
            )}
            </li>
          )
        })}
      </ul>
    </section>
  )
}

/** Yes/no attributes plus a note for the selected object. "✦" marks what the AI filled in. */
function AttributeEditor({ value, onChange }: { value: Attributes; onChange: (v: Attributes) => void }) {
  const ai = new Set(value.ai_suggested ?? [])
  const toggle = (k: 'occluded' | 'truncated') => {
    ai.delete(k) // the user decided: no longer an AI suggestion
    onChange({ ...value, [k]: !value[k], ai_suggested: [...ai] })
  }
  const chip = (k: 'occluded' | 'truncated', text: string, hint: string) => (
    <button type="button" title={hint} onClick={() => toggle(k)}
            className={`rounded-full border px-2 py-0.5 ${value[k]
              ? 'border-cyan-400/60 bg-cyan-500/15 text-cyan-200' : 'border-neutral-700 text-neutral-400 hover:text-neutral-200'}`}>
      {ai.has(k) && <span className="ai-text">✦ </span>}{text}
    </button>
  )
  return (
    <div className="ml-4 flex flex-wrap items-center gap-1.5 pb-1 text-[11px]" onClick={(e) => e.stopPropagation()}>
      {chip('occluded', 'Partly hidden', 'Another object hides part of it (occluded)')}
      {chip('truncated', 'Cut off at edge', 'Part of it is outside the photo (truncated)')}
      <input value={value.note ?? ''} maxLength={500} placeholder="Note (optional)"
             onChange={(e) => onChange({ ...value, note: e.target.value })}
             className="min-w-0 flex-1 rounded border border-neutral-700 bg-neutral-900 px-1.5 py-0.5 text-neutral-200 placeholder:text-neutral-600" />
    </div>
  )
}
