import type {
  AiStatus, Annotation, AutoLabelResult, BoxGeom, DetectResult, EffortEvent, ImageInfo, ImageStatus, Label,
  LabelingSession, Mode, NextImagesResult, PolygonGeom, Scorecard, Project, PromptPoint, SegmentResult, SimilarResult, Stats, UploadResult,
} from './types'

// Relative on purpose: resolves against the page URL, so a proxy prefix
// such as /user/<name>/proxy/8000/ is kept.
const BASE = 'api'

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, init)
  if (!res.ok) {
    let message = res.statusText
    try {
      const body = await res.json()
      message = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, message)
  }
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export const api = {
  listProjects: () => request<Project[]>('/projects'),
  createProject: (name: string) => request<Project>('/projects', json('POST', { name })),
  getProject: (id: number) => request<Project>(`/projects/${id}`),

  /** Delete all annotations (images stay); optionally the effort stats and the labels too. */
  resetProject: (projectId: number, clearStats: boolean, removeLabels: boolean) =>
    request<{ annotations_deleted: number; images_reset: number; events_deleted: number; labels_deleted: number }>(
      `/projects/${projectId}/reset`, json('POST', { clear_stats: clearStats, remove_labels: removeLabels })),

  listLabels: (projectId: number) => request<Label[]>(`/projects/${projectId}/labels`),
  createLabel: (projectId: number, name: string, color: string) =>
    request<Label>(`/projects/${projectId}/labels`, json('POST', { name, color })),
  deleteLabel: (labelId: number) => request<void>(`/labels/${labelId}`, { method: 'DELETE' }),

  listImages: (projectId: number) => request<ImageInfo[]>(`/projects/${projectId}/images`),
  uploadImages: (projectId: number, files: File[]) => {
    const form = new FormData()
    files.forEach((f) => form.append('files', f))
    return request<UploadResult>(`/projects/${projectId}/images`, { method: 'POST', body: form })
  },
  getImage: (imageId: number) => request<ImageInfo>(`/images/${imageId}`),
  imageUrl: (imageId: number) => `${BASE}/images/${imageId}/file`,
  /** JPEG with the saved annotations drawn on (download). */
  renderedUrl: (imageId: number) => `${BASE}/images/${imageId}/rendered`,
  setImageStatus: (imageId: number, status: ImageStatus) =>
    request<ImageInfo>(`/images/${imageId}`, json('PATCH', { status })),
  deleteImage: (imageId: number) => request<void>(`/images/${imageId}`, { method: 'DELETE' }),

  getAnnotations: (imageId: number) => request<Annotation[]>(`/images/${imageId}/annotations`),
  putAnnotations: (imageId: number, annotations: Annotation[]) =>
    request<Annotation[]>(`/images/${imageId}/annotations`, json('PUT', { annotations })),

  aiStatus: () => request<AiStatus>('/ai/status'),
  embed: (imageId: number) => request<{ latency_ms: number; cached: boolean }>(`/images/${imageId}/embed`, { method: 'POST' }),
  segment: (imageId: number, points: PromptPoint[], box: BoxGeom | null) =>
    request<SegmentResult>(`/images/${imageId}/segment`, json('POST', { points, box })),
  /** Regions in imageId that look like the example (box or polygon) marked on sourceImageId. */
  findSimilar: (imageId: number, body: { source_image_id: number; label_id: number; save: boolean;
                                         box?: BoxGeom; polygon?: PolygonGeom['points']; strictness?: number }) =>
    request<SimilarResult>(`/images/${imageId}/similar`, json('POST', body)),
  /** Boxes in imageId matching a typed name ("car"), OWLv2 text query. */
  textSearch: (imageId: number, body: { text: string; label_id: number; save: boolean; strictness?: number }) =>
    request<SimilarResult>(`/images/${imageId}/text-search`, json('POST', body)),
  undoSimilar: (ids: string[]) => request<{ deleted: number }>('/similar/undo', json('POST', { ids })),
  /** YOLO boxes; label names YOLO does not know are searched with OWLv2 (open vocabulary). */
  detect: (imageId: number, conf = 0.1, names: string[] = [], vocab = false) =>
    request<DetectResult>(`/images/${imageId}/detect`, json('POST', { conf, names, vocab })),
  autoLabelProject: (projectId: number, conf: number, skipLabeled: boolean) =>
    request<AutoLabelResult>(`/projects/${projectId}/auto-label`, json('POST', { conf, skip_labeled: skipLabeled })),

  /** Unlabeled images, most uncertain for the detector first ("label these next"). */
  nextImages: (projectId: number, limit = 5) =>
    request<NextImagesResult>(`/projects/${projectId}/next-images?limit=${limit}`, { method: 'POST' }),
  scorecard: (projectId: number) => request<Scorecard>(`/projects/${projectId}/scorecard`),
  scorecardUrl: (projectId: number) => `${BASE}/projects/${projectId}/scorecard?download=true`,

  createSession: (projectId: number, mode: Mode, participant: string) =>
    request<LabelingSession>(`/projects/${projectId}/sessions`, json('POST', { mode, participant })),
  postEvents: (projectId: number, events: EffortEvent[], keepalive = false) =>
    request<void>(`/projects/${projectId}/events`, { ...json('POST', { events }), keepalive }),
  stats: (projectId: number) => request<Stats>(`/projects/${projectId}/stats`),
  eventsCsvUrl: (projectId: number) => `${BASE}/projects/${projectId}/events.csv`,
  exportUrl: (projectId: number, format: 'coco' | 'yolo' | 'voc' | 'images', onlyDone: boolean) =>
    `${BASE}/projects/${projectId}/export?format=${format}&only_done=${onlyDone}`,
}
