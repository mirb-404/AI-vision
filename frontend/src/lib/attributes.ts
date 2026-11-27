import type { Annotation, Attributes } from '../api/types'
import { boundingBox } from './geometry'

const EDGE_PX = 2 // a box this close to the image border is cut off by the frame

/**
 * The AI's first guess for an object's attributes, the way industry tools pre-fill them:
 * - truncated: the object touches the image edge (part of it is outside the photo);
 * - occluded: another object covers 25-90 % of it. More than 90 % means it sits inside the other
 *   one (a face inside a person: a part, not hidden), so that does not count.
 * The user can switch either off; keys the AI set are listed in ai_suggested.
 */
export function suggestAttributes(a: Annotation, others: Annotation[], width: number, height: number): Attributes {
  const b = boundingBox(a)
  const area = b.w * b.h
  const truncated = b.x <= EDGE_PX || b.y <= EDGE_PX || b.x + b.w >= width - EDGE_PX || b.y + b.h >= height - EDGE_PX
  const occluded = area > 0 && others.some((o) => {
    if (o.id === a.id) return false
    const c = boundingBox(o)
    const ix = Math.max(0, Math.min(b.x + b.w, c.x + c.w) - Math.max(b.x, c.x))
    const iy = Math.max(0, Math.min(b.y + b.h, c.y + c.h) - Math.max(b.y, c.y))
    const share = (ix * iy) / area
    return share >= 0.25 && share <= 0.9
  })
  const ai_suggested = [truncated && 'truncated', occluded && 'occluded'].filter(Boolean) as ('truncated' | 'occluded')[]
  return { truncated, occluded, ai_suggested }
}

/** Add AI-suggested attributes to new annotations (kept as they are when already set). */
export function withAttributes(items: Annotation[], existing: Annotation[], width: number, height: number): Annotation[] {
  const all = [...existing, ...items]
  return items.map((a) => (a.attributes && Object.keys(a.attributes).length
    ? a
    : { ...a, attributes: suggestAttributes(a, all, width, height) }))
}
