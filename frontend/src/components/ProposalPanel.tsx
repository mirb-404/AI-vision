import { btnSecondary } from '../lib/ui'
import { useAnnotate, type Proposal } from '../store/annotate'

/**
 * The AI assistant panel (YOLO pre-labels). Simple view: two plain switches and "Is this a …?"
 * questions. Expert view adds the confidence slider and score percentages.
 */
export function ProposalPanel({ available, detail, detecting, simple, autoSuggest, onAutoSuggest, autoAccept,
                                onAutoAccept, onDetect, onAccept, onReject }: {
  available: boolean
  detail?: string
  detecting: boolean
  simple: boolean
  autoSuggest: boolean
  onAutoSuggest: (on: boolean) => void
  autoAccept: boolean
  onAutoAccept: (on: boolean) => void
  onDetect: () => void
  onAccept: (items: Proposal[]) => void
  onReject: (items: Proposal[]) => void
}) {
  const { proposals, confThreshold, setConfThreshold, selectedProposalId, selectProposal } = useAnnotate()
  const visible = proposals.filter((p) => p.score >= confThreshold).sort((a, b) => b.score - a.score)

  return (
    <section className="space-y-2">
      <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-neutral-400">
        <span className={detecting ? 'ai-sparkle ai-text' : 'ai-text'}>✦</span>
        {simple ? 'AI assistant' : 'AI assistant (YOLO)'}
      </h3>

      <Switch on={autoSuggest} disabled={!available} onChange={onAutoSuggest}
              title="Always on" hint="AI looks at every image you open" />
      <Switch on={autoAccept} disabled={!available} onChange={onAutoAccept}
              title="Mark sure objects for me" hint="Only unsure ones are asked about" />

      <button className={`relative w-full overflow-hidden rounded-md p-px text-sm disabled:opacity-50 ${detecting ? '' : 'bg-neutral-700'}`}
              onClick={onDetect} disabled={!available || detecting}
              title={available ? undefined : detail}>
        {detecting && <span className="ai-gradient absolute inset-0" />}
        <span className="relative flex items-center justify-center gap-1.5 rounded-[5px] bg-neutral-900 px-3 py-1.5 font-medium text-neutral-100 hover:bg-neutral-800">
          {detecting ? <><span className="ai-sparkle ai-text">✦</span><span className="ai-text">Finding objects…</span></>
                     : <>✦ {autoSuggest ? 'Look again' : 'Find objects in this image'}</>}
        </span>
      </button>
      {!available && (
        <p className="text-xs text-neutral-500">
          {simple ? 'The AI is not available right now. You can still mark objects yourself.' : `Unavailable: ${detail ?? 'status unknown'}`}
        </p>
      )}

      {proposals.length > 0 && (
        <>
          {!simple && (
            <label className="block text-xs text-neutral-400">
              Min confidence: <span className="text-neutral-200">{Math.round(confThreshold * 100)}%</span>
              <input type="range" min={0.05} max={0.95} step={0.05} value={confThreshold}
                     onChange={(e) => setConfThreshold(Number(e.target.value))} className="w-full accent-yellow-400" />
            </label>
          )}
          {visible.length > 0 && (
            <>
              <p className="text-xs text-neutral-300">
                {simple ? 'AI is not sure about these. Are they right?' : `${visible.length} suggestions`}
              </p>
              <div className="flex gap-1.5">
                <button className={`${btnSecondary} flex-1 !px-2 text-xs`} onClick={() => onAccept(visible)}>
                  {simple ? `✓ All right (${visible.length})` : `Accept ${visible.length}`}
                </button>
                <button className={`${btnSecondary} flex-1 !px-2 text-xs`} onClick={() => onReject(visible)}>
                  {simple ? `✕ All wrong` : `Reject ${visible.length}`}
                </button>
              </div>
            </>
          )}
          {!simple && (
            <p className="text-xs text-neutral-500">{proposals.length - visible.length} hidden below threshold.</p>
          )}
          <ul className="max-h-48 space-y-1 overflow-y-auto">
            {visible.map((p, i) => (
              <li key={p.id} style={{ animationDelay: `${Math.min(i, 12) * 40}ms` }}
                  onClick={() => selectProposal(p.id)}
                  className={`flex animate-fade-up cursor-pointer items-center gap-2 rounded px-2 py-1 text-xs ${
                    p.id === selectedProposalId ? 'bg-yellow-400/15' : 'hover:bg-neutral-800'}`}>
                <span className="flex-1 truncate">{simple ? `Is this a ${p.label}?` : p.label}</span>
                {!simple && <span className="text-neutral-400">{Math.round(p.score * 100)}%</span>}
                <button className="rounded px-1 text-emerald-400 hover:bg-emerald-400/10 hover:text-emerald-300"
                        title="Yes, keep it (A)" aria-label={`Yes, it is a ${p.label}`}
                        onClick={(e) => { e.stopPropagation(); onAccept([p]) }}>{simple ? '✓ Yes' : '✓'}</button>
                <button className="rounded px-1 text-red-400 hover:bg-red-400/10 hover:text-red-300"
                        title="No, remove it (Del)" aria-label="No, remove it"
                        onClick={(e) => { e.stopPropagation(); onReject([p]) }}>{simple ? '✕ No' : '✕'}</button>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  )
}

function Switch({ on, disabled, onChange, title, hint }: {
  on: boolean
  disabled: boolean
  onChange: (on: boolean) => void
  title: string
  hint: string
}) {
  const active = on && !disabled
  return (
    <label className={`flex items-center justify-between gap-2 rounded-md border border-neutral-800 px-2 py-1.5 text-xs text-neutral-300 ${
      disabled ? 'opacity-50' : 'cursor-pointer'}`}>
      <span>
        {title}
        <span className="block text-[11px] text-neutral-500">{hint}</span>
      </span>
      <input type="checkbox" role="switch" checked={on} disabled={disabled}
             onChange={(e) => onChange(e.target.checked)} className="peer sr-only" />
      <span aria-hidden className={`relative h-5 w-9 shrink-0 rounded-full transition-colors peer-focus-visible:ring-2 peer-focus-visible:ring-cyan-400 ${
        active ? 'ai-gradient' : 'bg-neutral-700'}`}>
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform ${
          active ? 'translate-x-[18px]' : 'translate-x-0.5'}`} />
      </span>
    </label>
  )
}
