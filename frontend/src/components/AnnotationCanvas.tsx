import type Konva from 'konva'
import type { KonvaEventObject } from 'konva/lib/Node'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Circle, Image as KImage, Layer, Line, Rect, Stage, Text, Transformer } from 'react-konva'
import type { BoxGeom, Label, PolygonGeom } from '../api/types'
import { effort } from '../lib/effort'
import { clamp, normalizeBox } from '../lib/geometry'
import { useAnnotate } from '../store/annotate'

interface Props {
  imageUrl: string
  imageWidth: number
  imageHeight: number
  labels: Label[]
  onCreate: (kind: 'box', geometry: BoxGeom, tool: string) => void
  onCreatePolygon: (geometry: PolygonGeom) => void
}

type Pt = { x: number; y: number }

const PROPOSAL_COLOR = '#facc15'
const SMART_COLOR = '#22d3ee'

export function AnnotationCanvas({ imageUrl, imageWidth, imageHeight, labels, onCreate, onCreatePolygon }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const stageRef = useRef<Konva.Stage>(null)
  const trRef = useRef<Konva.Transformer>(null)
  const smartDown = useRef<(Pt & { button: number }) | null>(null)

  const [size, setSize] = useState({ w: 800, h: 600 })
  const [img, setImg] = useState<HTMLImageElement | null>(null)
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 })
  const [drag, setDrag] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null)
  const [cursor, setCursor] = useState<Pt | null>(null)
  const [vertexDrag, setVertexDrag] = useState<{ id: string; idx: number; x: number; y: number } | null>(null)

  const s = useAnnotate()
  const { tool, annotations, selectedId, draftPolygon, proposals, selectedProposalId, confThreshold } = s
  const labelById = useMemo(() => new Map(labels.map((l) => [l.id, l])), [labels])
  const k = 1 / view.scale // converts screen pixels to image pixels for stroke widths, radii

  // Track container size.
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) =>
      setSize({ w: Math.max(100, entry.contentRect.width), h: Math.max(100, entry.contentRect.height) }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Load the image element.
  useEffect(() => {
    setImg(null)
    const el = new window.Image()
    el.onload = () => setImg(el)
    el.src = imageUrl
    return () => {
      el.onload = null
    }
  }, [imageUrl])

  const fit = useCallback(() => {
    const scale = Math.min(size.w / imageWidth, size.h / imageHeight) * 0.95
    setView({ scale, x: (size.w - imageWidth * scale) / 2, y: (size.h - imageHeight * scale) / 2 })
  }, [size.w, size.h, imageWidth, imageHeight])

  useEffect(fit, [fit, imageUrl])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input, textarea, select')) return
      if (e.key === 'f') fit()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [fit])

  // Attach the resize handles to the selected box.
  useEffect(() => {
    const tr = trRef.current
    if (!tr) return
    const sel = annotations.find((a) => a.id === selectedId)
    const node = sel?.kind === 'box' && tool === 'select' ? stageRef.current?.findOne(`#ann-${sel.id}`) : null
    tr.nodes(node ? [node] : [])
    tr.getLayer()?.batchDraw()
  }, [annotations, selectedId, tool])

  /** Pointer position in image pixels, clamped to the image. */
  const pointer = (): Pt | null => {
    const p = stageRef.current?.getRelativePointerPosition()
    if (!p) return null
    return { x: clamp(p.x, 0, imageWidth), y: clamp(p.y, 0, imageHeight) }
  }

  const onWheel = (e: KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault()
    const ptr = stageRef.current?.getPointerPosition()
    if (!ptr) return
    const scale = clamp(view.scale * (e.evt.deltaY > 0 ? 1 / 1.15 : 1.15), 0.05, 40)
    const mx = (ptr.x - view.x) / view.scale
    const my = (ptr.y - view.y) / view.scale
    setView({ scale, x: ptr.x - mx * scale, y: ptr.y - my * scale })
  }

  const isBackground = (target: Konva.Node) => target === stageRef.current || target.name() === 'background'

  const onMouseDown = (e: KonvaEventObject<MouseEvent>) => {
    if (tool === 'pan' || e.evt.button === 1) return
    effort.click()
    const p = pointer()
    if (!p) return
    const button = e.evt.button
    if (tool === 'box' && button === 0) {
      setDrag({ x1: p.x, y1: p.y, x2: p.x, y2: p.y })
    } else if (tool === 'polygon' && button === 0) {
      const pts = draftPolygon
      const nearFirst = pts.length >= 6 && Math.hypot(p.x - pts[0], p.y - pts[1]) < 10 * k
      if (nearFirst) {
        onCreatePolygon({ points: pts })
      } else {
        s.setDraftPolygon([...pts, p.x, p.y])
      }
    } else if (tool === 'smart') {
      smartDown.current = { ...p, button }
      if (button === 0) setDrag({ x1: p.x, y1: p.y, x2: p.x, y2: p.y })
    } else if (tool === 'select' && isBackground(e.target)) {
      s.select(null)
    }
  }

  const onMouseMove = () => {
    const p = pointer()
    if (!p) return
    if (tool === 'polygon') setCursor(p)
    if (drag) setDrag({ ...drag, x2: p.x, y2: p.y })
  }

  const onMouseUp = (e: KonvaEventObject<MouseEvent>) => {
    if (tool === 'box' && drag) {
      const b = normalizeBox(drag.x1, drag.y1, drag.x2, drag.y2)
      setDrag(null)
      if (b.w / k > 4 && b.h / k > 4) onCreate('box', b, 'box')
    } else if (tool === 'smart' && smartDown.current) {
      const down = smartDown.current
      smartDown.current = null
      const current = drag
      setDrag(null)
      const isBox = current && down.button === 0 &&
        (Math.abs(current.x2 - current.x1) / k > 5 || Math.abs(current.y2 - current.y1) / k > 5)
      if (isBox) {
        s.setSmartPrompt(s.smartPoints, normalizeBox(current.x1, current.y1, current.x2, current.y2))
      } else {
        const label = down.button === 2 || e.evt.altKey ? 0 : 1
        s.setSmartPrompt([...s.smartPoints, { x: down.x, y: down.y, label }], s.smartBox)
      }
    }
  }

  const onStageDragEnd = (e: KonvaEventObject<DragEvent>) => {
    if (e.target === stageRef.current) setView((v) => ({ ...v, x: e.target.x(), y: e.target.y() }))
  }

  const selectShape = (id: string) => (e: KonvaEventObject<MouseEvent>) => {
    if (tool !== 'select') return
    e.cancelBubble = true
    effort.click()
    s.select(id)
  }

  const visibleProposals = proposals.filter((p) => p.score >= confThreshold)
  const cursorStyle = tool === 'pan' ? 'grab' : tool === 'select' ? 'default' : 'crosshair'
  const fontSize = 12 * k

  return (
    <div ref={containerRef} className="relative h-full w-full overflow-hidden bg-slate-900" style={{ cursor: cursorStyle }}>
      <Stage
        ref={stageRef}
        width={size.w}
        height={size.h}
        x={view.x}
        y={view.y}
        scaleX={view.scale}
        scaleY={view.scale}
        draggable={tool === 'pan'}
        onDragEnd={onStageDragEnd}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onContextMenu={(e) => e.evt.preventDefault()}
      >
        <Layer>
          {img && <KImage image={img} width={imageWidth} height={imageHeight} name="background" />}
        </Layer>

        <Layer>
          {annotations.map((a) => {
            const color = labelById.get(a.label_id)?.color ?? '#94a3b8'
            const selected = a.id === selectedId
            const strokeWidth = (selected ? 3 : 2) * k
            const dash = a.source === 'ai' ? [6 * k, 3 * k] : undefined
            if (a.kind === 'box') {
              const g = a.geometry
              return (
                <Rect
                  key={a.id}
                  id={`ann-${a.id}`}
                  x={g.x} y={g.y} width={g.w} height={g.h}
                  stroke={color} strokeWidth={strokeWidth} dash={dash}
                  fill={color + (selected ? '44' : '22')}
                  listening={tool === 'select'}
                  draggable={tool === 'select' && selected}
                  onMouseDown={selectShape(a.id)}
                  onDragEnd={(e) => s.update(a.id, { geometry: { ...g, x: e.target.x(), y: e.target.y() } })}
                  onTransformEnd={(e) => {
                    const node = e.target
                    const w = Math.max(1, node.width() * node.scaleX())
                    const h = Math.max(1, node.height() * node.scaleY())
                    node.scale({ x: 1, y: 1 })
                    s.update(a.id, { geometry: { x: node.x(), y: node.y(), w, h } })
                  }}
                />
              )
            }
            let points = a.geometry.points
            if (vertexDrag?.id === a.id) {
              points = [...points]
              points[vertexDrag.idx * 2] = vertexDrag.x
              points[vertexDrag.idx * 2 + 1] = vertexDrag.y
            }
            return (
              <Line
                key={a.id}
                id={`ann-${a.id}`}
                points={points}
                closed
                stroke={color} strokeWidth={strokeWidth} dash={dash}
                fill={color + (selected ? '44' : '22')}
                listening={tool === 'select'}
                draggable={tool === 'select' && selected}
                onMouseDown={selectShape(a.id)}
                onDragEnd={(e) => {
                  const dx = e.target.x()
                  const dy = e.target.y()
                  e.target.position({ x: 0, y: 0 })
                  s.update(a.id, { geometry: { points: points.map((v, i) => v + (i % 2 ? dy : dx)) } })
                }}
              />
            )
          })}

          {/* Vertex handles of the selected polygon. Alt+click removes a vertex. */}
          {tool === 'select' && (() => {
            const sel = annotations.find((a) => a.id === selectedId)
            if (sel?.kind !== 'polygon') return null
            const pts = sel.geometry.points
            return Array.from({ length: pts.length / 2 }, (_, i) => (
              <Circle
                key={`v-${i}`}
                x={vertexDrag?.id === sel.id && vertexDrag.idx === i ? vertexDrag.x : pts[i * 2]}
                y={vertexDrag?.id === sel.id && vertexDrag.idx === i ? vertexDrag.y : pts[i * 2 + 1]}
                radius={5 * k}
                fill="#fff" stroke="#0f172a" strokeWidth={1.5 * k}
                draggable
                onMouseDown={(e) => {
                  e.cancelBubble = true
                  effort.click()
                  if (e.evt.altKey && pts.length > 6) {
                    s.update(sel.id, { geometry: { points: pts.filter((_, j) => Math.floor(j / 2) !== i) } })
                  }
                }}
                onDragMove={(e) => setVertexDrag({ id: sel.id, idx: i, x: e.target.x(), y: e.target.y() })}
                onDragEnd={(e) => {
                  const next = [...pts]
                  next[i * 2] = clamp(e.target.x(), 0, imageWidth)
                  next[i * 2 + 1] = clamp(e.target.y(), 0, imageHeight)
                  setVertexDrag(null)
                  s.update(sel.id, { geometry: { points: next } })
                }}
              />
            ))
          })()}

          <Transformer ref={trRef} rotateEnabled={false} keepRatio={false} ignoreStroke flipEnabled={false} />
        </Layer>

        {/* Detector proposals, AI previews and drafts. */}
        <Layer>
          {visibleProposals.map((p) => {
            const selected = p.id === selectedProposalId
            return (
              <Rect
                key={p.id}
                x={p.box.x} y={p.box.y} width={p.box.w} height={p.box.h}
                stroke={PROPOSAL_COLOR} strokeWidth={(selected ? 3 : 1.5) * k} dash={[8 * k, 4 * k]}
                fill={selected ? PROPOSAL_COLOR + '33' : undefined}
                listening={tool === 'select'}
                onMouseDown={(e) => {
                  if (tool !== 'select') return
                  e.cancelBubble = true
                  effort.click()
                  s.selectProposal(p.id)
                }}
              />
            )
          })}
          {visibleProposals.map((p) => (
            <Text
              key={`t-${p.id}`}
              x={p.box.x} y={p.box.y - fontSize * 1.3}
              text={`${p.label} ${(p.score * 100).toFixed(0)}%`}
              fontSize={fontSize} fill={PROPOSAL_COLOR} listening={false}
            />
          ))}

          {s.smartPreview && s.smartPreview.polygon.length >= 6 && (
            <Line points={s.smartPreview.polygon} closed stroke={SMART_COLOR} strokeWidth={2 * k}
                  fill={SMART_COLOR + '44'} listening={false} />
          )}
          {s.smartBox && (
            <Rect x={s.smartBox.x} y={s.smartBox.y} width={s.smartBox.w} height={s.smartBox.h}
                  stroke={SMART_COLOR} strokeWidth={1.5 * k} dash={[6 * k, 4 * k]} listening={false} />
          )}
          {s.smartPoints.map((p, i) => (
            <Circle key={`sp-${i}`} x={p.x} y={p.y} radius={5 * k} fill={p.label ? '#22c55e' : '#ef4444'}
                    stroke="#fff" strokeWidth={1.5 * k} listening={false} />
          ))}

          {drag && (
            <Rect x={Math.min(drag.x1, drag.x2)} y={Math.min(drag.y1, drag.y2)}
                  width={Math.abs(drag.x2 - drag.x1)} height={Math.abs(drag.y2 - drag.y1)}
                  stroke="#fff" strokeWidth={1.5 * k} dash={[6 * k, 4 * k]} listening={false} />
          )}

          {tool === 'polygon' && draftPolygon.length > 0 && (
            <>
              <Line points={cursor ? [...draftPolygon, cursor.x, cursor.y] : draftPolygon}
                    stroke="#fff" strokeWidth={1.5 * k} listening={false} />
              {Array.from({ length: draftPolygon.length / 2 }, (_, i) => (
                <Circle key={`d-${i}`} x={draftPolygon[i * 2]} y={draftPolygon[i * 2 + 1]}
                        radius={(i === 0 ? 6 : 4) * k} fill={i === 0 ? '#22c55e' : '#fff'} listening={false} />
              ))}
            </>
          )}
        </Layer>
      </Stage>

      <div className="absolute bottom-3 right-3 flex gap-2 text-xs">
        <span className="rounded bg-slate-800/80 px-2 py-1 text-slate-300">{Math.round(view.scale * 100)}%</span>
        <button onClick={fit} className="rounded bg-slate-800/80 px-2 py-1 text-slate-200 hover:bg-slate-700">
          Fit (F)
        </button>
      </div>
      {!img && (
        <div className="absolute inset-0 flex items-center justify-center text-slate-400">Loading image…</div>
      )}
    </div>
  )
}
