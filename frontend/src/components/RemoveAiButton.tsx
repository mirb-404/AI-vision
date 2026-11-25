import { useEffect, useState } from 'react'

/**
 * Top-right corner of the canvas: remove every label the AI added to this image (untouched AI
 * labels only, not ones the user fixed). After removing, offers Undo for a few seconds.
 */
export function RemoveAiButton({ count, onRemove, onUndo }: {
  count: number
  onRemove: () => number
  onUndo: () => void
}) {
  const [removed, setRemoved] = useState(0)
  useEffect(() => {
    if (!removed) return
    const t = window.setTimeout(() => setRemoved(0), 10000)
    return () => window.clearTimeout(t)
  }, [removed])

  if (removed) {
    return (
      <div role="status"
           className="absolute right-3 top-3 z-20 flex animate-fade-in items-center gap-2 rounded-lg border border-neutral-700 bg-neutral-900/95 px-3 py-1.5 text-xs text-neutral-200 shadow-lg shadow-black/50 backdrop-blur">
        Removed {removed} AI label{removed === 1 ? '' : 's'}
        <button className="font-medium text-white underline-offset-2 hover:underline"
                onClick={() => { onUndo(); setRemoved(0) }}>Undo</button>
      </div>
    )
  }
  if (!count) return null
  return (
    <button data-tour="remove-ai" onClick={() => setRemoved(onRemove())}
            title="Remove all labels the AI added to this image. Labels you made or fixed stay."
            className="absolute right-3 top-3 z-20 flex animate-fade-in items-center gap-1.5 rounded-lg border border-neutral-700 bg-neutral-900/90 px-3 py-1.5 text-xs text-neutral-200 shadow-lg shadow-black/50 backdrop-blur transition-colors hover:border-red-400/60 hover:text-white">
      <span className="ai-text">✦</span> Remove AI labels ({count})
    </button>
  )
}
