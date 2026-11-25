/** What the "find alike objects" card shows: an offer, a running search, or the result. */
export type SimilarState =
  | { stage: 'offer'; name: string }
  | { stage: 'running'; name: string; done: number; total: number; found: number }
  | { stage: 'result'; name: string; found: number; images: number; stopped: boolean }

/**
 * Card at the bottom of the canvas. After the user marks an object (e.g. one cricket ball), it
 * offers to mark every alike object in the project; then shows progress and an Undo.
 */
export function SimilarSearch({ state, onFind, onStop, onUndo, onClose }: {
  state: SimilarState
  onFind: () => void
  onStop: () => void
  onUndo: () => void
  onClose: () => void
}) {
  const name = `“${state.name}”`
  return (
    <div role="status"
         className="absolute bottom-4 left-1/2 z-20 w-[min(26rem,92%)] -translate-x-1/2 animate-fade-up rounded-xl border border-neutral-700/80 bg-neutral-900/95 p-3 text-sm shadow-2xl shadow-black/60 backdrop-blur">
      {state.stage === 'offer' && (
        <div className="flex items-center gap-3">
          <span className="ai-text text-base">✦</span>
          <span className="min-w-0 flex-1 text-neutral-200">Mark every {name} in all images?</span>
          <button className="rounded-md bg-white px-3 py-1 font-medium text-neutral-950 hover:bg-neutral-200" onClick={onFind}>
            Find all
          </button>
          <button className="text-neutral-500 hover:text-neutral-100" onClick={onClose} aria-label="Not now">✕</button>
        </div>
      )}

      {state.stage === 'running' && (
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <span className="ai-sparkle ai-text text-base">✦</span>
            <span className="min-w-0 flex-1 text-neutral-200">
              Looking for {name}… image {Math.min(state.done + 1, state.total)} of {state.total}
              {state.found > 0 && <span className="text-neutral-400"> · {state.found} found</span>}
            </span>
            <button className="text-neutral-400 hover:text-neutral-100" onClick={onStop}>Stop</button>
          </div>
          <div className="h-1 overflow-hidden rounded-full bg-neutral-800">
            <div className="ai-gradient h-full rounded-full transition-[width] duration-700 ease-out"
                 style={{ width: `${(state.done / Math.max(state.total, 1)) * 100}%` }} />
          </div>
        </div>
      )}

      {state.stage === 'result' && (
        <div className="flex items-center gap-3">
          <span className="ai-text text-base">✦</span>
          <span className="min-w-0 flex-1 text-neutral-200">
            {state.found
              ? `Marked ${state.found} more ${name} in ${state.images} image${state.images === 1 ? '' : 's'}${state.stopped ? ' (stopped)' : ''}.`
              : `No other ${name} found.`}
          </span>
          {state.found > 0 && (
            <button className="rounded-md border border-neutral-600 px-3 py-1 text-neutral-200 hover:bg-neutral-800" onClick={onUndo}>
              Undo
            </button>
          )}
          <button className="text-neutral-500 hover:text-neutral-100" onClick={onClose} aria-label="Close">✕</button>
        </div>
      )}
    </div>
  )
}
