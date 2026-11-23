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
