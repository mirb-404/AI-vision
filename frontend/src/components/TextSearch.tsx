import { useState } from 'react'

/**
 * "Find by name": type what to look for ("car", "helmet"), the AI marks every one in all images.
 * Works for names the auto-label model does not know (OWLv2 open-vocabulary detection).
 */
export function TextSearch({ busy, onSearch }: { busy: boolean; onSearch: (text: string) => void }) {
  const [text, setText] = useState('')
  const submit = () => {
    const t = text.trim()
    if (!t || busy) return
    onSearch(t)
    setText('')
  }
  return (
    <section data-tour="text-search" className="space-y-1.5">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
        <span className="ai-text">✦</span> Find by name
      </h3>
      <form className="flex gap-1.5" onSubmit={(e) => { e.preventDefault(); submit() }}>
        <input value={text} onChange={(e) => setText(e.target.value)} maxLength={60}
               placeholder="e.g. car, helmet, ball"
               aria-label="What should the AI find?"
               className="min-w-0 flex-1 rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-sm text-neutral-100 placeholder:text-neutral-500 focus:border-neutral-400 focus:outline-none" />
        <button type="submit" disabled={busy || !text.trim()}
                className="rounded-md bg-white px-3 py-1 text-sm font-medium text-neutral-950 hover:bg-neutral-200 disabled:opacity-40">
          Find
        </button>
      </form>
      <p className="text-xs text-neutral-500">The AI marks every one in all images. Undo is on the card.</p>
    </section>
  )
}
