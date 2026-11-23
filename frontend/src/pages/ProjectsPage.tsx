import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router'
import { api } from '../api/client'
import type { Project } from '../api/types'
import { btnPrimary, card, input } from '../lib/ui'

export function ProjectsPage() {
  const [projects, setProjects] = useState<Project[] | null>(null)
  const [name, setName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    api.listProjects().then(setProjects).catch((e) => setError(e.message))
  }, [])

  const create = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) return
    try {
      const p = await api.createProject(name.trim())
      navigate(`/projects/${p.id}`)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Projects</h1>
        <p className="text-sm text-slate-400">A project holds images, label classes and their annotations.</p>
      </div>

      <form onSubmit={create} className="flex gap-2">
        <input className={`${input} w-72`} placeholder="New project name" value={name}
               onChange={(e) => setName(e.target.value)} maxLength={200} />
        <button className={btnPrimary} disabled={!name.trim()}>Create project</button>
      </form>

      {error && <p className="text-sm text-red-400">{error}</p>}

      {projects === null ? (
        <p className="text-slate-400">Loading…</p>
      ) : projects.length === 0 ? (
        <p className="text-slate-400">No projects yet. Create one above.</p>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {projects.map((p) => (
            <Link key={p.id} to={`/projects/${p.id}`} className={`${card} p-4 transition hover:border-indigo-500`}>
              <div className="font-medium">{p.name}</div>
              <div className="mt-2 text-sm text-slate-400">
                {p.image_count} images · {p.done_count} done · {p.annotation_count} annotations
              </div>
              <div className="mt-3 h-1.5 overflow-hidden rounded bg-slate-800">
                <div className="h-full bg-emerald-500"
                     style={{ width: `${p.image_count ? (100 * p.done_count) / p.image_count : 0}%` }} />
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
