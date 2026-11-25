import { useEffect, useRef, useState } from 'react'

/**
 * Overlay on the canvas while a model works: rotating gradient ring, soft halo, a light band
 * sweeping over the image and a status pill. Never blocks input (pointer-events: none).
 */
export function AiWorking({ label }: { label: string | null }) {
  // Keep the last label during the fade-out so the pill does not go blank.
  const [shown, setShown] = useState(label)
  useEffect(() => {
    if (label) setShown(label)
  }, [label])
  const active = !!label

  return (
    <div aria-hidden={!active}
         className={`pointer-events-none absolute inset-0 z-10 transition-opacity duration-700 ease-in-out ${active ? 'opacity-100' : 'opacity-0'}`}>
      {(active || shown) && (
        <>
          <div className="ai-gradient ai-halo absolute inset-0" />
          <div className="ai-gradient ai-ring absolute inset-0" />
          <div className="ai-sweep absolute inset-[3px]" />
          <div role="status"
               className="absolute bottom-4 left-1/2 flex -translate-x-1/2 items-center gap-2 rounded-full border border-white/10 bg-neutral-950/85 px-3.5 py-1.5 shadow-lg shadow-black/50 backdrop-blur">
            <span className="ai-sparkle ai-text text-sm">✦</span>
            <span className="ai-text text-xs font-semibold">{shown}</span>
          </div>
        </>
      )}
    </div>
  )
}

/** Short toast when the always-on assistant has new suggestions for this image. */
export function AiSuggestionsToast({ added, review, nonce }: { added: number; review: number; nonce: number }) {
  const first = useRef(nonce)
  if (nonce === first.current) return null
  return (
    <div key={nonce} role="status"
         className="pointer-events-none absolute left-1/2 top-14 z-20 animate-toast-soft rounded-full p-px shadow-xl shadow-black/50">
      <div className="ai-gradient absolute inset-0 rounded-full" />
      <div className="relative flex items-center gap-2 rounded-full bg-neutral-950 px-3.5 py-1.5 text-xs">
        <span className="ai-sparkle ai-text">✦</span>
        <span className="text-neutral-100">
          {toastText(added, review)}
        </span>
      </div>
    </div>
  )
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`

function toastText(added: number, review: number): string {
  if (added && review) return `AI marked ${plural(added, 'object')}. ${review} more need your check`
  if (added) return `AI marked ${plural(added, 'object')} for you. Check they look right`
  if (review) return `AI has ${plural(review, 'suggestion')} for you to check`
  return 'AI found nothing new here. Tap objects to mark them'
}
