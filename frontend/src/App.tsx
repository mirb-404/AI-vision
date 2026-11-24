import { Link, Route, Routes, useLocation } from 'react-router'
import { AnnotatePage } from './pages/AnnotatePage'
import { ProjectPage } from './pages/ProjectPage'
import { ProjectsPage } from './pages/ProjectsPage'
import { StatsPage } from './pages/StatsPage'

function Shell({ children }: { children: React.ReactNode }) {
  const { pathname } = useLocation()
  return (
    <div className="min-h-full bg-[radial-gradient(ellipse_at_top,rgba(255,255,255,0.06),transparent_60%)]">
      <header className="sticky top-0 z-20 border-b border-neutral-800/80 bg-neutral-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-3">
          <Link to="/" className="flex items-center gap-2 font-display text-lg font-semibold tracking-tight text-white">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-white text-sm text-neutral-950">✦</span>
            AI Annotator
          </Link>
          <span className="hidden text-xs text-neutral-500 sm:inline">AI-assisted image labeling</span>
        </div>
      </header>
      {/* Keyed by path so each page fades in on navigation. */}
      <main key={pathname} className="mx-auto max-w-6xl animate-fade-up px-4 py-6">{children}</main>
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
      <Route path="*" element={<Shell><p className="text-neutral-400">Page not found. <Link className="text-neutral-200 underline underline-offset-4" to="/">Go home</Link></p></Shell>} />
    </Routes>
  )
}
