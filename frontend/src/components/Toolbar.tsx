import type { Tool } from '../store/annotate'

const TOOLS: { id: Tool; key: string; label: string; icon: string }[] = [
  { id: 'select', key: 'V', label: 'Select / edit', icon: '↖' },
  { id: 'box', key: 'B', label: 'Bounding box', icon: '▭' },
  { id: 'polygon', key: 'P', label: 'Polygon', icon: '⬠' },
  { id: 'smart', key: 'S', label: 'Smart select (SAM)', icon: '✦' },
  { id: 'pan', key: 'H', label: 'Pan', icon: '✋' },
]

export function Toolbar({ tool, onTool, smartEnabled, onUndo, onRedo, canUndo, canRedo }: {
  tool: Tool
  onTool: (t: Tool) => void
  smartEnabled: boolean
  onUndo: () => void
  onRedo: () => void
  canUndo: boolean
  canRedo: boolean
}) {
  const base = 'flex h-10 w-10 items-center justify-center rounded-md text-lg transition disabled:opacity-30'
  return (
    <nav className="flex w-14 shrink-0 flex-col items-center gap-1 border-r border-slate-800 py-2">
      {TOOLS.map((t) => {
        const disabled = t.id === 'smart' && !smartEnabled
        return (
          <button
            key={t.id}
            title={disabled ? `${t.label}: needs AI mode and SAM weights` : `${t.label} (${t.key})`}
            disabled={disabled}
            onClick={() => onTool(t.id)}
            className={`${base} ${tool === t.id ? 'bg-indigo-600 text-white' : 'text-slate-300 hover:bg-slate-800'} ${
              t.id === 'smart' && !disabled && tool !== t.id ? 'text-cyan-300' : ''}`}
          >
            {t.icon}
          </button>
        )
      })}
      <div className="my-2 h-px w-8 bg-slate-800" />
      <button title="Undo (Ctrl+Z)" onClick={onUndo} disabled={!canUndo} className={`${base} text-slate-300 hover:bg-slate-800`}>↶</button>
      <button title="Redo (Ctrl+Y)" onClick={onRedo} disabled={!canRedo} className={`${base} text-slate-300 hover:bg-slate-800`}>↷</button>
    </nav>
  )
}
