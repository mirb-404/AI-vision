import { ApiError, api } from '../api/client'
import type { Annotation, EffortEvent, EventKind } from '../api/types'
import { vertexCount } from './geometry'

/**
 * Effort logger for the manual-vs-AI study.
 *
 * Time per object = time between consecutive commits on the open image
 * (the first object is timed from image open). It therefore includes looking,
 * drawing and any corrections, in both modes alike. Clicks = pointer presses
 * on the canvas in the same interval.
 */
class EffortTracker {
  private queue: EffortEvent[] = []
  private projectId: number | null = null
  private sessionId: number | null = null
  private imageId: number | null = null
  private objectStart = performance.now()
  private clicks = 0
  private timer: number | undefined

  start(projectId: number, sessionId: number) {
    this.flush()
    this.projectId = projectId
    this.sessionId = sessionId
    window.clearInterval(this.timer)
    this.timer = window.setInterval(() => this.flush(), 3000)
  }

  stop() {
    this.flush(true)
    window.clearInterval(this.timer)
    this.sessionId = null
  }

  openImage(imageId: number) {
    this.imageId = imageId
    this.resetObject()
    this.log('image_opened')
  }

  resetObject() {
    this.objectStart = performance.now()
    this.clicks = 0
  }

  click() {
    this.clicks += 1
  }

  log(kind: EventKind, payload: Record<string, unknown> = {}, annotationId: string | null = null) {
    if (this.sessionId == null) return
    this.queue.push({
      session_id: this.sessionId,
      image_id: this.imageId,
      annotation_id: annotationId,
      kind,
      ts: new Date().toISOString(),
      payload,
    })
  }

  /** Log new finished objects. A batch (e.g. "accept all") shares the elapsed time and clicks equally. */
  commit(annotations: Annotation[], tool: string) {
    if (!annotations.length) return
    const n = annotations.length
    const duration = performance.now() - this.objectStart
    for (const a of annotations) {
      this.log('annotation_committed', {
        duration_ms: Math.round(duration / n),
        clicks: this.clicks / n,
        vertices: vertexCount(a),
        tool,
        kind: a.kind,
        source: a.source,
        score: a.score,
        label_id: a.label_id,
      }, a.id)
    }
    this.resetObject()
  }

  flush(keepalive = false) {
    if (!this.queue.length || this.projectId == null) return
    const batch = this.queue.splice(0)
    api.postEvents(this.projectId, batch, keepalive).catch((err) => {
      // Keep events for the next try unless the server rejected them as invalid.
      if (!(err instanceof ApiError && err.status === 422)) this.queue.unshift(...batch)
    })
  }
}

export const effort = new EffortTracker()

window.addEventListener('pagehide', () => effort.flush(true))
