import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

/** One tutorial stop: highlights the element with data-tour={target}. */
export interface TourStep {
  target: string
  title: string
  body: string
}

type Spot = { x: number; y: number; r: number }

const PAD = 10
const MARGIN = 12
const CARD_MAX_W = 340
const CARD_EST_H = 210

function findTarget(target: string): HTMLElement | null {
  const el = document.querySelector<HTMLElement>(`[data-tour="${target}"]`)
  if (!el) return null
  const r = el.getBoundingClientRect()
  return r.width > 0 && r.height > 0 ? el : null
}

/** Circle around the element; very large targets (the canvas) get a capped spotlight at their centre. */
function spotFor(el: HTMLElement): Spot {
  const r = el.getBoundingClientRect()
  const cap = Math.min(window.innerWidth, window.innerHeight) / 2.6
  return {
    x: r.left + r.width / 2,
    y: r.top + r.height / 2,
    r: Math.max(26, Math.min(Math.max(r.width, r.height) / 2 + PAD, cap)),
  }
}

/**
 * Guided tour with a moving circular spotlight. Steps whose target is missing or
 * hidden (e.g. a panel not shown in the current mode) are skipped.
 */
export function Tour({ steps, open, onClose }: { steps: TourStep[]; open: boolean; onClose: () => void }) {
  const [index, setIndex] = useState(0)
  const [spot, setSpot] = useState<Spot | null>(null)
  const [viewport, setViewport] = useState({ w: window.innerWidth, h: window.innerHeight })
  const nextRef = useRef<HTMLButtonElement>(null)

  const available = open ? steps.filter((s) => findTarget(s.target)) : []
  const step = available[Math.min(index, available.length - 1)]

  useEffect(() => {
    if (open) setIndex(0)
  }, [open])

  const measure = useCallback(() => {
    setViewport({ w: window.innerWidth, h: window.innerHeight })
    const el = step && findTarget(step.target)
    setSpot(el ? spotFor(el) : null)
  }, [step])

  // Bring the target into view, then place the spotlight on it.
  useLayoutEffect(() => {
    if (!open || !step) return
    const el = findTarget(step.target)
    el?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
    measure()
    const id = window.setTimeout(measure, 120) // after smooth layout changes settle
    return () => window.clearTimeout(id)
  }, [open, step, measure])

  useEffect(() => {
    if (!open) return
    window.addEventListener('resize', measure)
    window.addEventListener('scroll', measure, true)
    return () => {
      window.removeEventListener('resize', measure)
      window.removeEventListener('scroll', measure, true)
    }
  }, [open, measure])

  const next = useCallback(() => {
    if (index >= available.length - 1) onClose()
    else setIndex(index + 1)
  }, [index, available.length, onClose])
  const back = useCallback(() => setIndex((i) => Math.max(0, i - 1)), [])

  // Capture-phase keys so the page's own shortcuts don't fire while the tour is open.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Tab') return // keep keyboard focus moving between the card's buttons
      if (e.key === 'Escape') onClose()
      else if (e.key === 'ArrowRight' || e.key === 'Enter') next()
      else if (e.key === 'ArrowLeft') back()
      e.preventDefault()
      e.stopImmediatePropagation()
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [open, next, back, onClose])

  useEffect(() => {
    if (open) nextRef.current?.focus({ preventScroll: true })
  }, [open, index])

  if (!open || !step) return null

  // Card on the half of the screen the spotlight is not in; horizontally near the target.
  const cardW = Math.min(CARD_MAX_W, viewport.w - 2 * MARGIN)
  const s = spot ?? { x: viewport.w / 2, y: viewport.h / 2, r: 0 }
  const left = Math.min(Math.max(s.x - cardW / 2, MARGIN), viewport.w - cardW - MARGIN)
  const below = s.y < viewport.h / 2
  const maxOffset = viewport.h - CARD_EST_H - MARGIN
  const cardPos = below
    ? { top: Math.min(s.y + s.r + 16, maxOffset) }
    : { bottom: Math.min(viewport.h - (s.y - s.r - 16), maxOffset) }

  // Portal to <body>: an ancestor with a transform (page fade-in) would otherwise make
  // `position: fixed` relative to that ancestor instead of the viewport.
  return createPortal(
    <div className="fixed inset-0 z-50 animate-fade-in" role="dialog" aria-modal="true" aria-labelledby="tour-title">
      {/* Blocks clicks on the page while the tour is open. */}
      <div className="absolute inset-0" onClick={(e) => e.stopPropagation()} />

      {/* Spotlight: a circle whose huge shadow darkens everything else. */}
      <div
        aria-hidden
        className="pointer-events-none absolute rounded-full transition-all duration-500 ease-[cubic-bezier(0.22,1,0.36,1)]"
        style={{
          left: s.x - s.r,
          top: s.y - s.r,
          width: s.r * 2,
          height: s.r * 2,
          boxShadow: '0 0 0 9999px rgba(0,0,0,0.74), 0 0 0 2px rgba(255,255,255,0.9), 0 0 32px 6px rgba(255,255,255,0.25)',
        }}
      >
        <span className="absolute inset-0 animate-spot-pulse rounded-full border-2 border-white/60" />
      </div>

      <div
        key={step.target}
        className="absolute animate-fade-up rounded-xl border border-neutral-700 bg-neutral-900/95 p-4 text-sm shadow-2xl shadow-black/60 backdrop-blur"
        style={{ left, width: cardW, ...cardPos }}
      >
        <div className="mb-1 flex items-center justify-between gap-2">
          <h3 id="tour-title" className="text-base font-semibold text-white">{step.title}</h3>
          <span className="font-mono text-xs text-neutral-500">
            {Math.min(index, available.length - 1) + 1}/{available.length}
          </span>
        </div>
        <p className="leading-relaxed text-neutral-300" aria-live="polite">{step.body}</p>
        <div className="mt-3 flex items-center gap-1">
          {available.map((st, i) => (
            <span key={st.target}
                  className={`h-1.5 rounded-full transition-all duration-300 ${i === index ? 'w-5 bg-white' : 'w-1.5 bg-neutral-600'}`} />
          ))}
        </div>
        <div className="mt-3 flex items-center gap-2">
          <button className="mr-auto text-xs text-neutral-400 hover:text-white" onClick={onClose}>Skip tour</button>
          <button className="rounded-lg px-3 py-1.5 text-neutral-300 hover:bg-neutral-800 disabled:opacity-30"
                  onClick={back} disabled={index === 0}>Back</button>
          <button ref={nextRef} className="rounded-lg bg-white px-3 py-1.5 font-medium text-neutral-950 hover:bg-neutral-200"
                  onClick={next}>
            {index >= available.length - 1 ? 'Finish' : 'Next'}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  )
}

/** Open state for a tour; opens by itself on the first visit (remembered per key in localStorage). */
export function useTour(storageKey: string, ready: boolean) {
  const [open, setOpen] = useState(false)
  useEffect(() => {
    if (!ready) return
    let seen = false
    try {
      seen = localStorage.getItem(storageKey) === '1'
    } catch {
      /* storage blocked: just don't auto-open */
      seen = true
    }
    if (!seen) {
      const id = window.setTimeout(() => setOpen(true), 400)
      return () => window.clearTimeout(id)
    }
  }, [storageKey, ready])

  const close = useCallback(() => {
    setOpen(false)
    try {
      localStorage.setItem(storageKey, '1')
    } catch {
      /* ignore */
    }
  }, [storageKey])

  return { open, start: () => setOpen(true), close }
}

/** Round "?" button that restarts a tour. */
export function HelpButton({ onClick }: { onClick: () => void }) {
  return (
    <button data-tour="help" onClick={onClick} title="Show the tutorial" aria-label="Show the tutorial"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-neutral-700 text-sm font-semibold text-neutral-300 transition hover:border-neutral-400 hover:text-white active:scale-95">
      ?
    </button>
  )
}
