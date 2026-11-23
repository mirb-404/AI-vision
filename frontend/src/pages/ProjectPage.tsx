import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'
import { api } from '../api/client'
import type { AiStatus, ImageInfo, Label, Mode, Project } from '../api/types'
import { nextColor } from '../lib/geometry'
import { loadSettings, saveSettings, type SessionSettings } from '../lib/settings'
import { btnDanger, btnPrimary, btnSecondary, card, input, statusStyle } from '../lib/ui'

export function ProjectPage() {
  const projectId = Number(useParams().projectId)
  const navigate = useNavigate()
  const [project, setProject] = useState<Project | null>(null)
  const [labels, setLabels] = useState<Label[]>([])
  const [images, setImages] = useState<ImageInfo[]>([])
  const [aiStatus, setAiStatus] = useState<AiStatus | null>(null)
  const [settings, setSettings] = useState<SessionSettings>(() => loadSettings(projectId))
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [onlyDone, setOnlyDone] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  const refresh = useCallback(async () => {
    try {
      const [p, l, i] = await Promise.all([api.getProject(projectId), api.listLabels(projectId), api.listImages(projectId)])
      setProject(p)
      setLabels(l)
      setImages(i)
    } catch (e) {
      setError((e as Error).message)
    }
  }, [projectId])

  useEffect(() => {
    refresh()
    api.aiStatus().then(setAiStatus).catch(() => setAiStatus(null))
  }, [refresh])

  const updateSettings = (next: SessionSettings) => {
    setSettings(next)
    saveSettings(projectId, next)
  }

  const upload = async (files: File[]) => {
    const imagesOnly = files.filter((f) => f.type.startsWith('image/'))
    if (!imagesOnly.length) return
    setUploading(true)
    setError(null)
    try {
      const res = await api.uploadImages(projectId, imagesOnly)
      if (res.errors.length) setError(res.errors.map((e) => `${e.filename}: ${e.error}`).join('; '))
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setUploading(false)
    }
  }

  const removeImage = async (img: ImageInfo) => {
    if (!window.confirm(`Delete ${img.filename} and its ${img.annotation_count} annotation(s)?`)) return
    try {
      await api.deleteImage(img.id)
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const startLabeling = () => {
    const next = images.find((i) => i.status !== 'done') ?? images[0]
    if (next) navigate(`/projects/${projectId}/annotate/${next.id}`)
  }

  if (!project) return <p className="text-slate-400">{error ?? 'Loading…'}</p>

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Link to="/" className="text-sm text-slate-400 hover:text-slate-200">← Projects</Link>
          <h1 className="text-2xl font-semibold">{project.name}</h1>
          <p className="text-sm text-slate-400">
            {project.image_count} images · {project.done_count} done · {project.annotation_count} annotations
          </p>
        </div>
        <div className="flex gap-2">
          <Link to={`/projects/${projectId}/stats`} className={btnSecondary}>Effort stats</Link>
          <button className={btnPrimary} onClick={startLabeling} disabled={!images.length || !labels.length}
                  title={!labels.length ? 'Add at least one label first' : undefined}>
            Start labeling
          </button>
        </div>
      </div>

      {error && <p className="rounded-md bg-red-500/10 px-3 py-2 text-sm text-red-300">{error}</p>}

      <div className="grid gap-4 lg:grid-cols-3">
        <LabelManager projectId={projectId} labels={labels} onChange={refresh} onError={setError} />

        <section className={`${card} space-y-3 p-4`}>
          <h2 className="font-medium">Labeling session</h2>
          <div className="flex rounded-md border border-slate-700 p-0.5 text-sm">
            {(['ai', 'manual'] as Mode[]).map((m) => (
              <button key={m} onClick={() => updateSettings({ ...settings, mode: m })}
                      className={`flex-1 rounded px-2 py-1 ${settings.mode === m ? 'bg-indigo-600 text-white' : 'text-slate-300'}`}>
                {m === 'ai' ? 'AI-assisted' : 'Manual only'}
              </button>
            ))}
          </div>
          <input className={`${input} w-full`} placeholder="Participant id (for the study, optional)"
                 value={settings.participant} maxLength={100}
                 onChange={(e) => updateSettings({ ...settings, participant: e.target.value })} />
          <p className="text-xs text-slate-500">
            Time, clicks and corrections are logged per session to compare manual and AI-assisted labeling.
          </p>
          <div className="space-y-1 text-xs">
            <ModelBadge name="SAM (click → mask)" status={aiStatus?.sam} />
            <ModelBadge name="YOLO (auto boxes)" status={aiStatus?.yolo} />
          </div>
        </section>

        <section className={`${card} space-y-3 p-4`}>
          <h2 className="font-medium">Export</h2>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input type="checkbox" checked={onlyDone} onChange={(e) => setOnlyDone(e.target.checked)} />
            Only images marked done
          </label>
          <div className="flex flex-wrap gap-2">
            <a className={btnSecondary} href={api.exportUrl(projectId, 'coco', onlyDone)}>COCO JSON (.zip)</a>
            <a className={btnSecondary} href={api.exportUrl(projectId, 'yolo', onlyDone)}>YOLO (.zip)</a>
          </div>
          <p className="text-xs text-slate-500">Both include the images. COCO keeps polygons; YOLO uses boxes.</p>
        </section>
      </div>

      <section
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); upload(Array.from(e.dataTransfer.files)) }}
        className={`${card} flex flex-col items-center justify-center gap-2 border-dashed p-6 text-center ${dragOver ? 'border-indigo-500 bg-indigo-500/10' : ''}`}
      >
        <p className="text-sm text-slate-300">Drop images here (JPEG, PNG, WEBP, BMP)</p>
        <button className={btnSecondary} onClick={() => fileInput.current?.click()} disabled={uploading}>
          {uploading ? 'Uploading…' : 'Choose files'}
        </button>
        <input ref={fileInput} type="file" accept="image/*" multiple hidden
               onChange={(e) => { upload(Array.from(e.target.files ?? [])); e.target.value = '' }} />
      </section>

      {images.length > 0 && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
          {images.map((img) => (
            <div key={img.id} className={`${card} group overflow-hidden`}>
              <Link to={`/projects/${projectId}/annotate/${img.id}`} className="block aspect-square bg-slate-800">
                <img src={api.imageUrl(img.id)} alt={img.filename} loading="lazy" className="h-full w-full object-cover" />
              </Link>
              <div className="space-y-1 p-2 text-xs">
                <div className="truncate text-slate-300" title={img.filename}>{img.filename}</div>
                <div className="flex items-center justify-between">
                  <span className={`rounded px-1.5 py-0.5 ${statusStyle[img.status]}`}>{img.status.replace('_', ' ')}</span>
                  <span className="text-slate-500">{img.annotation_count} ann.</span>
                </div>
                <button className={`${btnDanger} w-full !py-0.5 text-xs opacity-0 group-hover:opacity-100`}
                        onClick={() => removeImage(img)}>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function ModelBadge({ name, status }: { name: string; status?: { available: boolean; detail: string } }) {
  const ok = status?.available
  return (
    <div className="flex items-center gap-2">
      <span className={`h-2 w-2 rounded-full ${ok ? 'bg-emerald-400' : 'bg-slate-600'}`} />
      <span className="text-slate-300">{name}</span>
      <span className="truncate text-slate-500" title={status?.detail}>{status ? status.detail : 'unknown'}</span>
    </div>
  )
}

function LabelManager({ projectId, labels, onChange, onError }: {
  projectId: number
  labels: Label[]
  onChange: () => void
  onError: (msg: string) => void
}) {
  const [name, setName] = useState('')
  const [color, setColor] = useState(() => nextColor(labels.map((l) => l.color)))

  useEffect(() => setColor(nextColor(labels.map((l) => l.color))), [labels])

  const add = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) return
    try {
      await api.createLabel(projectId, name.trim(), color)
      setName('')
      onChange()
    } catch (err) {
      onError((err as Error).message)
    }
  }

  const remove = async (label: Label) => {
    try {
      await api.deleteLabel(label.id)
      onChange()
    } catch (err) {
      onError((err as Error).message)
    }
  }

  return (
    <section className={`${card} space-y-3 p-4`}>
      <h2 className="font-medium">Labels</h2>
      <form onSubmit={add} className="flex gap-2">
        <input type="color" value={color} onChange={(e) => setColor(e.target.value)}
               className="h-8 w-9 cursor-pointer rounded border border-slate-700 bg-transparent" />
        <input className={`${input} min-w-0 flex-1`} placeholder="e.g. car" value={name} maxLength={100}
               onChange={(e) => setName(e.target.value)} />
        <button className={btnSecondary} disabled={!name.trim()}>Add</button>
      </form>
      {labels.length === 0 ? (
        <p className="text-xs text-slate-500">
          No labels yet. Tip: use COCO class names (person, car, dog…) so YOLO proposals map to them.
        </p>
      ) : (
        <ul className="space-y-1">
          {labels.map((l, i) => (
            <li key={l.id} className="flex items-center gap-2 text-sm">
              <span className="h-3 w-3 rounded-sm" style={{ background: l.color }} />
              <span className="flex-1">{l.name}</span>
              {i < 9 && <kbd className="text-xs text-slate-500">{i + 1}</kbd>}
              <button className="text-xs text-slate-500 hover:text-red-400" onClick={() => remove(l)}>✕</button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
