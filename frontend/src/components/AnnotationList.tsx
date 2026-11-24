import type { Label, Source } from '../api/types'
import { useAnnotate } from '../store/annotate'

const SOURCE_BADGE: Record<Source, { text: string; cls: string }> = {
  manual: { text: 'manual', cls: 'bg-neutral-700 text-neutral-200' },
  ai: { text: 'AI', cls: 'bg-cyan-500/20 text-cyan-300' },
  ai_edited: { text: 'AI · edited', cls: 'bg-amber-500/20 text-amber-300' },
}

export function AnnotationList({ labels, samEnabled }: { labels: Label[]; samEnabled: boolean }) {
  const { annotations, selectedId, select, update, remove, startRefine } = useAnnotate()
  const colorOf = (id: number) => labels.find((l) => l.id === id)?.color ?? '#94a3b8'

  return (
    <section className="space-y-1.5">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
        Annotations ({annotations.length})
      </h3>
      {annotations.length === 0 && <p className="text-xs text-neutral-500">Nothing labeled on this image yet.</p>}
      <ul className="space-y-1">
        {annotations.map((a) => {
          const badge = SOURCE_BADGE[a.source]
          return (
            <li key={a.id} onClick={() => { select(a.id); useAnnotate.getState().setTool('select') }}
                className={`flex cursor-pointer items-center gap-1.5 rounded px-1.5 py-1 text-xs ${
                  a.id === selectedId ? 'bg-neutral-800 ring-1 ring-neutral-600' : 'hover:bg-neutral-900'}`}>
              <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: colorOf(a.label_id) }} />
              <select value={a.label_id} onClick={(e) => e.stopPropagation()}
                      onChange={(e) => update(a.id, { label_id: Number(e.target.value) })}
                      className="min-w-0 flex-1 rounded border border-neutral-700 bg-neutral-900 px-1 py-0.5">
                {labels.map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
              </select>
              <span className="text-neutral-500">{a.kind === 'box' ? '▭' : '⬠'}</span>
              <span className={`rounded px-1 py-0.5 ${badge.cls}`}>{badge.text}</span>
              {a.kind === 'box' && samEnabled && (
                <button title="Refine box into a mask with SAM" className="text-cyan-300 hover:text-cyan-200"
                        onClick={(e) => { e.stopPropagation(); startRefine(a.id, a.geometry) }}>✦</button>
              )}
              <button title="Delete" className="text-neutral-500 hover:text-red-400"
                      onClick={(e) => { e.stopPropagation(); remove(a.id) }}>✕</button>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
