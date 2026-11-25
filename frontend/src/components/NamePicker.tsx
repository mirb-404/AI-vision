import { useState } from 'react'
import type { Label } from '../api/types'

/**
 * "What is this?" card shown after drawing a shape the AI could not name. Lets a beginner
 * name objects as they go instead of setting up labels first. Enter picks the last-used name.
 */
export function NamePicker({ labels, lastId, onPick, onCancel }: {
  labels: Label[]
  lastId: number | null
  onPick: (choice: { id: number } | { name: string }) => void
  onCancel: () => void
}) {
  const [text, setText] = useState('')
  const ordered = [...labels].sort((a, b) => Number(b.id === lastId) - Number(a.id === lastId))
  const typed = text.trim()
  const submit = () => {
    if (typed) onPick({ name: typed })
    else if (lastId != null) onPick({ id: lastId })
  }

  return (
    <div role="dialog" aria-label="Name this object"
         className="absolute left-1/2 top-3 z-30 w-[min(22rem,92%)] -translate-x-1/2 animate-fade-up rounded-xl border border-neutral-700 bg-neutral-900/95 p-3 shadow-2xl shadow-black/60 backdrop-blur">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-display text-sm font-semibold text-white">What is this?</h3>
        <button className="text-neutral-500 hover:text-neutral-100" onClick={onCancel} aria-label="Discard shape">✕</button>
      </div>
      {ordered.length > 0 && (
        <div className="mb-2 flex max-h-28 flex-wrap gap-1.5 overflow-y-auto">
          {ordered.map((l) => (
            <button key={l.id} onClick={() => onPick({ id: l.id })}
                    className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs hover:bg-neutral-800 ${
                      l.id === lastId ? 'border-neutral-300 text-white' : 'border-neutral-700 text-neutral-300'}`}>
              <span className="h-2 w-2 rounded-full" style={{ background: l.color }} />
              {l.name}
            </button>
          ))}
        </div>
      )}
      <form className="flex gap-1.5" onSubmit={(e) => { e.preventDefault(); submit() }}>
        <input autoFocus value={text} maxLength={60} onChange={(e) => setText(e.target.value)}
               onKeyDown={(e) => e.key === 'Escape' && onCancel()}
               placeholder={ordered.length ? 'Or type a new name…' : 'Type a name, e.g. dog'}
               className="min-w-0 flex-1 rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm outline-none focus:border-neutral-400" />
        <button type="submit" disabled={!typed && lastId == null}
                className="rounded-md bg-white px-3 text-sm font-medium text-neutral-950 disabled:opacity-40">
          {typed ? 'Add' : 'OK'}
        </button>
      </form>
    </div>
  )
}
