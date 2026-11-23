import { Link, Route, Routes } from 'react-router'
import { AnnotatePage } from './pages/AnnotatePage'
import { ProjectPage } from './pages/ProjectPage'
import { ProjectsPage } from './pages/ProjectsPage'
import { StatsPage } from './pages/StatsPage'

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-full">
      <header className="border-b border-slate-800 bg-slate-950/80">
        <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-3">
          <Link to="/" className="font-semibold tracking-tight text-slate-100">
            AI Annotator
          </Link>
          <span className="text-xs text-slate-500">AI-assisted image labeling</span>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">{children}</main>
    </div>
  )
}

export function App() {
  return (
    <Routes>
      <Route path="/" element={<Shell><ProjectsPage /></Shell>} />
      <Route path="/projects/:projectId" element={<Shell><ProjectPage /></Shell>} />
      <Route path="/projects/:projectId/stats" element={<Shell><StatsPage /></Shell>} />
      {/* Annotation view is full-screen, no shell. */}
      <Route path="/projects/:projectId/annotate/:imageId" element={<AnnotatePage />} />
      <Route path="*" element={<Shell><p className="text-slate-400">Page not found. <Link className="text-indigo-400" to="/">Go home</Link></p></Shell>} />
    </Routes>
  )
}
