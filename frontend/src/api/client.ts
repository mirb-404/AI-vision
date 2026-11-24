import type {
  AiStatus, Annotation, BoxGeom, DetectResult, EffortEvent, ImageInfo, ImageStatus, Label,
  LabelingSession, Mode, Project, PromptPoint, SegmentResult, Stats, UploadResult,
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
  detect: (imageId: number, conf = 0.1) => request<DetectResult>(`/images/${imageId}/detect`, json('POST', { conf })),

  createSession: (projectId: number, mode: Mode, participant: string) =>
    request<LabelingSession>(`/projects/${projectId}/sessions`, json('POST', { mode, participant })),
  postEvents: (projectId: number, events: EffortEvent[], keepalive = false) =>
    request<void>(`/projects/${projectId}/events`, { ...json('POST', { events }), keepalive }),
  stats: (projectId: number) => request<Stats>(`/projects/${projectId}/stats`),
  eventsCsvUrl: (projectId: number) => `${BASE}/projects/${projectId}/events.csv`,
  exportUrl: (projectId: number, format: 'coco' | 'yolo', onlyDone: boolean) =>
    `${BASE}/projects/${projectId}/export?format=${format}&only_done=${onlyDone}`,
}
