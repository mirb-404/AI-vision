import type { Annotation, BoxGeom } from '../api/types'

/** Random id that also works on plain-HTTP origins (crypto.randomUUID needs a secure context). */
export function uid(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(12))
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

export const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi)

export function normalizeBox(x1: number, y1: number, x2: number, y2: number): BoxGeom {
  return { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1) }
}

/** Axis-aligned bounding box of a box or polygon annotation. */
export function boundingBox(a: Annotation): BoxGeom {
  if (a.kind === 'box') return a.geometry
  const p = a.geometry.points
  const xs = p.filter((_, i) => i % 2 === 0)
  const ys = p.filter((_, i) => i % 2 === 1)
  return normalizeBox(Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys))
}

export function boxIou(a: BoxGeom, b: BoxGeom): number {
  const iw = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x))
  const ih = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y))
  const inter = iw * ih
  const union = a.w * a.h + b.w * b.h - inter
  return union > 0 ? inter / union : 0
}

export function vertexCount(a: Annotation): number {
  return a.kind === 'box' ? 2 : a.geometry.points.length / 2
}

export const LABEL_COLORS = [
  '#ef4444', '#3b82f6', '#22c55e', '#eab308', '#a855f7',
  '#f97316', '#06b6d4', '#ec4899', '#84cc16', '#6366f1',
]

export function nextColor(used: string[]): string {
  return LABEL_COLORS.find((c) => !used.includes(c)) ?? LABEL_COLORS[used.length % LABEL_COLORS.length]
}
