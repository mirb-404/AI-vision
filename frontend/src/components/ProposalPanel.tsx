import { btnSecondary } from '../lib/ui'
import { useAnnotate, type Proposal } from '../store/annotate'

/** Detector pre-labels: run YOLO, filter by confidence, accept or reject each box. */
export function ProposalPanel({ available, detail, detecting, onDetect, onAccept, onReject }: {
  available: boolean
  detail?: string
  detecting: boolean
  onDetect: () => void
  onAccept: (items: Proposal[]) => void
  onReject: (items: Proposal[]) => void
}) {
  const { proposals, confThreshold, setConfThreshold, selectedProposalId, selectProposal } = useAnnotate()
  const visible = proposals.filter((p) => p.score >= confThreshold).sort((a, b) => b.score - a.score)

  return (
    <section className="space-y-2">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">AI pre-labels (YOLO)</h3>
      <button className={`${btnSecondary} w-full`} onClick={onDetect} disabled={!available || detecting}
              title={available ? undefined : detail}>
        {detecting ? 'Detecting…' : 'Auto-label this image'}
      </button>
      {!available && <p className="text-xs text-slate-500">Unavailable: {detail ?? 'status unknown'}</p>}

      {proposals.length > 0 && (
        <>
          <label className="block text-xs text-slate-400">
            Min confidence: <span className="text-slate-200">{Math.round(confThreshold * 100)}%</span>
            <input type="range" min={0.05} max={0.95} step={0.05} value={confThreshold}
                   onChange={(e) => setConfThreshold(Number(e.target.value))} className="w-full accent-yellow-400" />
          </label>
          <div className="flex gap-1.5">
            <button className={`${btnSecondary} flex-1 !px-2 text-xs`} disabled={!visible.length}
                    onClick={() => onAccept(visible)}>Accept {visible.length}</button>
            <button className={`${btnSecondary} flex-1 !px-2 text-xs`} disabled={!visible.length}
                    onClick={() => onReject(visible)}>Reject {visible.length}</button>
          </div>
          <p className="text-xs text-slate-500">
            {proposals.length - visible.length} hidden below threshold.
          </p>
          <ul className="max-h-48 space-y-1 overflow-y-auto">
            {visible.map((p) => (
              <li key={p.id}
                  onClick={() => selectProposal(p.id)}
                  className={`flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-xs ${
                    p.id === selectedProposalId ? 'bg-yellow-400/15' : 'hover:bg-slate-800'}`}>
                <span className="flex-1 truncate">{p.label}</span>
                <span className="text-slate-400">{Math.round(p.score * 100)}%</span>
                <button className="text-emerald-400 hover:text-emerald-300" title="Accept (A)"
                        onClick={(e) => { e.stopPropagation(); onAccept([p]) }}>✓</button>
                <button className="text-red-400 hover:text-red-300" title="Reject (Del)"
                        onClick={(e) => { e.stopPropagation(); onReject([p]) }}>✕</button>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  )
}
