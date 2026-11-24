import type { Tool } from '../store/annotate'

/** Tool metadata, shared by the toolbar, the tool-change toast and the tutorial. */
export const TOOLS: { id: Tool; key: string; label: string; icon: string; tip: string }[] = [
  { id: 'select', key: 'V', label: 'Select / edit', icon: '↖', tip: 'Tap a shape to select it, then drag, resize or change its label.' },
  { id: 'box', key: 'B', label: 'Bounding box', icon: '▭', tip: 'Drag on the image to draw a rectangle around an object.' },
  { id: 'polygon', key: 'P', label: 'Polygon', icon: '⬠', tip: 'Tap points around the outline; tap the first point or press Enter to close.' },
  { id: 'smart', key: 'S', label: 'Smart select (SAM)', icon: '✦', tip: 'Tap an object or drag a box and SAM traces its mask for you.' },
  { id: 'pan', key: 'H', label: 'Pan', icon: '✋', tip: 'Drag to move around the image. Use + / − or the wheel to zoom.' },
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
  const base = 'flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-lg transition duration-200 ease-out disabled:opacity-30'
  return (
    <nav className="flex shrink-0 items-center gap-1 overflow-x-auto border-b border-neutral-800 bg-neutral-950 px-2 py-1 md:w-14 md:flex-col md:border-b-0 md:border-r md:px-0 md:py-2">
      {TOOLS.map((t) => {
        const disabled = t.id === 'smart' && !smartEnabled
        const active = tool === t.id
        return (
          <button
            key={t.id}
            data-tour={`tool-${t.id}`}
            title={disabled ? `${t.label}: needs AI mode and SAM weights` : `${t.label} (${t.key})`}
            aria-label={t.label}
            aria-pressed={active}
            disabled={disabled}
            onClick={() => onTool(t.id)}
            className={`${base} ${active
              ? 'animate-tool-pop bg-white text-neutral-950 shadow-md shadow-white/10'
              : 'text-neutral-400 hover:bg-neutral-800 hover:text-neutral-100'} ${
              t.id === 'smart' && !disabled && !active ? '!text-cyan-300' : ''}`}
          >
            {t.icon}
          </button>
        )
      })}
      <div className="mx-1 h-8 w-px shrink-0 bg-neutral-800 md:mx-0 md:my-2 md:h-px md:w-8" />
      <div data-tour="history" className="flex gap-1 md:flex-col">
        <button title="Undo (Ctrl+Z)" aria-label="Undo" onClick={onUndo} disabled={!canUndo}
                className={`${base} text-neutral-400 hover:bg-neutral-800 hover:text-neutral-100`}>↶</button>
        <button title="Redo (Ctrl+Y)" aria-label="Redo" onClick={onRedo} disabled={!canRedo}
                className={`${base} text-neutral-400 hover:bg-neutral-800 hover:text-neutral-100`}>↷</button>
      </div>
    </nav>
  )
}
