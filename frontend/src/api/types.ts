// Mirrors backend/app/schemas.py. Keep both in sync.

export type Mode = 'manual' | 'ai'
export type Source = 'manual' | 'ai' | 'ai_edited'
export type ImageStatus = 'new' | 'in_progress' | 'done'

export interface Project {
  id: number
  name: string
  created_at: string
  image_count: number
  done_count: number
  annotation_count: number
}

export interface Label {
  id: number
  project_id: number
  name: string
  color: string
}

export interface ImageInfo {
  id: number
  project_id: number
  filename: string
  width: number
  height: number
  status: ImageStatus
  annotation_count: number
}

export interface UploadResult {
  uploaded: ImageInfo[]
  errors: { filename: string; error: string }[]
}

export interface BoxGeom {
  x: number
  y: number
  w: number
  h: number
}

export interface PolygonGeom {
  points: number[] // [x0, y0, x1, y1, ...] in image pixels
}

interface AnnotationBase {
  id: string
  label_id: number
  source: Source
  score: number | null
}

export type BoxAnnotation = AnnotationBase & { kind: 'box'; geometry: BoxGeom }
export type PolygonAnnotation = AnnotationBase & { kind: 'polygon'; geometry: PolygonGeom }
export type Annotation = BoxAnnotation | PolygonAnnotation

export interface PromptPoint {
  x: number
  y: number
  label: 0 | 1
}

export interface SegmentResult {
  polygon: number[]
  score: number
  latency_ms: number
  embed_ms: number
}

export interface Detection {
  label: string
  class_id: number
  score: number
  box: BoxGeom
}

export interface DetectResult {
  detections: Detection[]
  latency_ms: number
}

export interface AutoLabelResult {
  images_labeled: number
  images_skipped: number
  annotations_created: number
  labels_created: string[]
  latency_ms: number
}

export interface ModelStatus {
  available: boolean
  loaded: boolean
  detail: string
}

export interface AiStatus {
  sam: ModelStatus
  yolo: ModelStatus
  /** OWLv2: find every object of the same kind as a marked example. */
  owl?: ModelStatus
  /** OWLv2 text tower: find every object matching a typed name. */
  owl_text?: ModelStatus
}

export interface LabelingSession {
  id: number
  project_id: number
  mode: Mode
  participant: string
  started_at: string
}

export type EventKind =
  | 'image_opened'
  | 'image_done'
  | 'annotation_committed'
  | 'annotation_edited'
  | 'annotation_deleted'
  | 'proposal_accepted'
  | 'proposal_rejected'

export interface EffortEvent {
  session_id: number | null
  image_id: number | null
  annotation_id?: string | null
  kind: EventKind
  ts: string
  payload: Record<string, unknown>
}

export interface Summary {
  n: number
  median: number | null
  q1: number | null
  q3: number | null
  mean: number | null
}

export interface ModeStats {
  objects: number
  seconds_per_object: Summary
  clicks_per_object: Summary
  by_tool: Record<string, number>
  sessions: number
}

export interface Stats {
  per_mode: Record<Mode, ModeStats>
  speed_up_median: number | null
  proposals: {
    accepted: number
    rejected: number
    edited_after_accept: number
    deleted_after_accept: number
    reviewed: number
    accept_rate: number | null
    edit_rate: number | null
  }
  inference_ms: Record<string, Summary>
}

/** One object found by the "find alike objects" search. id is set when the server saved it. */
export interface SimilarRegion {
  polygon: number[]
  box: BoxGeom
  score: number
  id: string | null
}

export interface SimilarResult {
  regions: SimilarRegion[]
  latency_ms: number
}

export interface NextImage {
  image_id: number
  /** 0 = the AI is sure about everything it sees, 1 = unsure or sees nothing it knows. */
  uncertainty: number
  detections: number
  unsure: number
}

export interface NextImagesResult {
  images: NextImage[]
  ranked: number
  latency_ms: number
}

export interface ScoreCounts {
  ai_kept: number
  ai_fixed: number
  ai_removed: number
  hand: number
  precision: number | null
  recall: number | null
  exact_share: number | null
}

export interface Scorecard {
  project: string
  images: number
  images_scope: string
  total: ScoreCounts
  per_label: Record<string, ScoreCounts>
  effort: {
    seconds_per_object: { manual: number | null; ai: number | null }
    speed_up_median: number | null
    accept_rate: number | null
  }
}
