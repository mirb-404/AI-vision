import { useState } from 'react'
import { Link } from 'react-router'
import { api } from '../api/client'
import type { NextImagesResult } from '../api/types'
import { btnSecondary, card } from '../lib/ui'

/**
 * "Label these next": the AI ranks unlabeled images by how unsure it is about them (uncertainty
 * sampling). Labeling those first teaches a fine-tuned model the most per image.
 */
export function NextImages({ projectId, available }: { projectId: number; available: boolean }) {
  const [result, setResult] = useState<NextImagesResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const rank = async () => {
    setBusy(true)
    setError(null)
    try {
      setResult(await api.nextImages(projectId, 6))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const reason = (detections: number, unsure: number) =>
    detections === 0 ? 'AI sees nothing it knows'
      : unsure ? `AI unsure about ${unsure} object${unsure === 1 ? '' : 's'}`
      : 'AI fairly sure'

  return (
    <section data-tour="next-images" className={`${card} space-y-3 p-4`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="font-medium"><span className="ai-text">✦</span> Label these next</h2>
          <p className="text-xs text-neutral-500">The AI picks the new images it is least sure about. Labeling them first helps it learn fastest.</p>
        </div>
        <button className={btnSecondary} onClick={() => void rank()} disabled={busy || !available}
                title={available ? undefined : 'Needs the auto-label model'}>
          {busy ? 'AI is looking…' : result ? 'Ask again' : 'Ask the AI'}
        </button>
      </div>
      {error && <p className="text-xs text-red-400">{error}</p>}
      {result && result.images.length === 0 && (
        <p className="text-xs text-neutral-400">No new images left. Everything has been started.</p>
      )}
      {result && result.images.length > 0 && (
        <div className="grid animate-fade-up grid-cols-3 gap-2 sm:grid-cols-6">
          {result.images.map((r, i) => (
            <Link key={r.image_id} to={`/projects/${projectId}/annotate/${r.image_id}`}
                  className="group overflow-hidden rounded-md border border-neutral-800 hover:border-neutral-500">
              <div className="relative aspect-square bg-neutral-800">
                <img src={api.imageUrl(r.image_id)} alt="" loading="lazy" className="h-full w-full object-cover" />
                <span className="absolute left-1 top-1 rounded bg-neutral-950/80 px-1.5 text-xs text-neutral-200">{i + 1}</span>
              </div>
              <div className="p-1.5 text-[11px] leading-tight text-neutral-400">{reason(r.detections, r.unsure)}</div>
            </Link>
          ))}
        </div>
      )}
    </section>
  )
}
