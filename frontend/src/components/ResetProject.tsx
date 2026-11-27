import { useState } from 'react'
import { api } from '../api/client'
import { clearAiCleared } from '../lib/aiCleared'
import { btnSecondary, card } from '../lib/ui'

/**
 * "Start over": delete every annotation of the project but keep the images, so testing again
 * needs no re-upload. Asks for confirmation first, because it cannot be undone.
 */
export function ResetProject({ projectId, annotationCount, onDone }: {
  projectId: number
  annotationCount: number
  onDone: () => void
}) {
  const [confirming, setConfirming] = useState(false)
  const [clearStats, setClearStats] = useState(false)
  const [removeLabels, setRemoveLabels] = useState(false)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

  const reset = async () => {
    setBusy(true)
    try {
      const r = await api.resetProject(projectId, clearStats, removeLabels)
      clearAiCleared(projectId)
      const n = r.images_reset
      setMessage(`Removed ${r.annotations_deleted} annotations; ${n} image${n === 1 ? ' is' : 's are'} new again`
        + (clearStats ? `, ${r.events_deleted} stats events cleared` : '')
        + (removeLabels ? `, ${r.labels_deleted} labels removed` : '') + '.')
      setConfirming(false)
      onDone()
    } catch (e) {
      setMessage((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section data-tour="reset" className={`${card} space-y-3 p-4`}>
      <h2 className="font-medium">Start over</h2>
      <p className="text-xs text-neutral-500">
        Remove all annotations and start labeling again. Your images stay, so nothing needs to be uploaded again.
      </p>
      {!confirming ? (
        <button className={`${btnSecondary} w-full hover:border-red-400/60 hover:text-red-300`}
                onClick={() => { setConfirming(true); setMessage(null) }} disabled={busy}>
          Reset all annotations
        </button>
      ) : (
        <div className="animate-fade-up space-y-2 rounded-md border border-red-500/40 bg-red-500/5 p-3 text-sm">
          <p className="text-red-200">
            Delete all {annotationCount} annotations of this project? This cannot be undone.
          </p>
          <label className="flex items-center gap-2 text-neutral-300">
            <input type="checkbox" checked={clearStats} onChange={(e) => setClearStats(e.target.checked)} />
            Also clear the effort stats (time and clicks log)
          </label>
          <label className="flex items-center gap-2 text-neutral-300">
            <input type="checkbox" checked={removeLabels} onChange={(e) => setRemoveLabels(e.target.checked)} />
            Also remove the labels
          </label>
          <div className="flex gap-2">
            <button className="rounded-md bg-red-600 px-3 py-1.5 font-medium text-white hover:bg-red-500 disabled:opacity-50"
                    onClick={() => void reset()} disabled={busy}>
              {busy ? 'Deleting…' : 'Yes, delete them'}
            </button>
            <button className={btnSecondary} onClick={() => setConfirming(false)} disabled={busy}>Cancel</button>
          </div>
        </div>
      )}
      {message && <p className="animate-fade-up text-xs text-emerald-300">{message}</p>}
    </section>
  )
}
