import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router'
import { api } from '../api/client'
import type { Mode, Scorecard, ScoreCounts, Stats, Summary } from '../api/types'
import { btnSecondary, card, fmt } from '../lib/ui'

const pct = (v: number | null) => (v == null ? '–' : `${Math.round(v * 100)}%`)
const range = (s: Summary, digits = 1) => (s.n ? `${fmt(s.q1, digits)}–${fmt(s.q3, digits)}` : '–')

export function StatsPage() {
  const projectId = Number(useParams().projectId)
  const [stats, setStats] = useState<Stats | null>(null)
  const [card_, setCard] = useState<Scorecard | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.stats(projectId).then(setStats).catch((e) => setError(e.message))
    api.scorecard(projectId).then(setCard).catch(() => setCard(null))
  }, [projectId])

  if (error) return <p className="text-red-400">{error}</p>
  if (!stats) return <p className="text-neutral-400">Loading…</p>

  const modes: Mode[] = ['manual', 'ai']
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <Link to={`/projects/${projectId}`} className="text-sm text-neutral-400 hover:text-neutral-200">← Project</Link>
          <h1 className="text-2xl font-semibold">Effort: manual vs AI-assisted</h1>
          <p className="max-w-2xl text-sm text-neutral-400">
            Time per object is the time between consecutive finished objects on an image (the first is timed from
            opening the image), including corrections. Compare at matched label quality before claiming a saving.
          </p>
        </div>
        <a className={btnSecondary} href={api.eventsCsvUrl(projectId)}>Download raw events (CSV)</a>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <Kpi title="Speed-up (median s/object)" value={stats.speed_up_median ? `${stats.speed_up_median.toFixed(2)}×` : '–'}
             hint="manual median ÷ AI median" />
        <Kpi title="Proposal accept rate" value={pct(stats.proposals.accept_rate)}
             hint={`${stats.proposals.accepted} accepted / ${stats.proposals.reviewed} reviewed`} />
        <Kpi title="Accepted AI labels later edited" value={pct(stats.proposals.edit_rate)}
             hint={`${stats.proposals.edited_after_accept} edits, ${stats.proposals.deleted_after_accept} deletions`} />
      </div>

      <section className={`${card} overflow-x-auto`}>
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase text-neutral-400">
            <tr className="border-b border-neutral-800">
              <th className="p-3">Mode</th><th className="p-3">Sessions</th><th className="p-3">Objects</th>
              <th className="p-3">Median s/object</th><th className="p-3">IQR</th>
              <th className="p-3">Median clicks/object</th><th className="p-3">Objects by tool</th>
            </tr>
          </thead>
          <tbody>
            {modes.map((m) => {
              const s = stats.per_mode[m]
              return (
                <tr key={m} className="border-b border-neutral-800/60 last:border-0">
                  <td className="p-3 font-medium">{m === 'ai' ? 'AI-assisted' : 'Manual'}</td>
                  <td className="p-3">{s.sessions}</td>
                  <td className="p-3">{s.objects}</td>
                  <td className="p-3">{fmt(s.seconds_per_object.median)}</td>
                  <td className="p-3 text-neutral-400">{range(s.seconds_per_object)}</td>
                  <td className="p-3">{fmt(s.clicks_per_object.median)}</td>
                  <td className="p-3 text-neutral-400">
                    {Object.entries(s.by_tool).map(([t, n]) => `${t}: ${n}`).join(', ') || '–'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </section>

      {card_ && <ScorecardSection projectId={projectId} data={card_} />}

      <section className={`${card} overflow-x-auto`}>
        <h2 className="p-3 pb-0 font-medium">Model latency (server-side, ms)</h2>
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase text-neutral-400">
            <tr className="border-b border-neutral-800">
              <th className="p-3">Model / stage</th><th className="p-3">Calls</th><th className="p-3">Median</th>
              <th className="p-3">IQR</th><th className="p-3">Mean</th>
            </tr>
          </thead>
          <tbody>
            {Object.keys(stats.inference_ms).length === 0 && (
              <tr><td className="p-3 text-neutral-500" colSpan={5}>No model calls yet.</td></tr>
            )}
            {Object.entries(stats.inference_ms).map(([k, s]) => (
              <tr key={k} className="border-b border-neutral-800/60 last:border-0">
                <td className="p-3">{k}</td><td className="p-3">{s.n}</td><td className="p-3">{fmt(s.median)}</td>
                <td className="p-3 text-neutral-400">{range(s)}</td><td className="p-3">{fmt(s.mean)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}

/** AI labels judged against the user's final labels: kept, fixed, removed, and what the user drew. */
function ScorecardSection({ projectId, data }: { projectId: number; data: Scorecard }) {
  const rows: [string, ScoreCounts][] = [['All labels', data.total], ...Object.entries(data.per_label)]
  return (
    <section className={`${card} overflow-x-auto`}>
      <div className="flex flex-wrap items-start justify-between gap-2 p-3 pb-0">
        <div>
          <h2 className="font-medium"><span className="ai-text">✦</span> AI scorecard</h2>
          <p className="max-w-2xl text-xs text-neutral-500">
            The AI&apos;s labels compared with your final labels on {data.images} image{data.images === 1 ? '' : 's'} ({data.images_scope}).
            Precision: AI labels you kept. Recall: objects the AI found. Exactly right: kept with no change.
          </p>
        </div>
        <a className={btnSecondary} href={api.scorecardUrl(projectId)}>Download scorecard (JSON)</a>
      </div>
      <table className="w-full text-sm">
        <thead className="text-left text-xs uppercase text-neutral-400">
          <tr className="border-b border-neutral-800">
            <th className="p-3">Label</th><th className="p-3">Kept</th><th className="p-3">Fixed</th>
            <th className="p-3">Removed</th><th className="p-3">Drawn by you</th>
            <th className="p-3">Precision</th><th className="p-3">Recall</th><th className="p-3">Exactly right</th>
          </tr>
        </thead>
        <tbody>
          {data.images === 0 && (
            <tr><td className="p-3 text-neutral-500" colSpan={8}>No AI labels reviewed yet.</td></tr>
          )}
          {data.images > 0 && rows.map(([name, c], i) => (
            <tr key={name} className={`border-b border-neutral-800/60 last:border-0 ${i === 0 ? 'font-medium' : ''}`}>
              <td className="p-3">{name}</td><td className="p-3">{c.ai_kept}</td><td className="p-3">{c.ai_fixed}</td>
              <td className="p-3">{c.ai_removed}</td><td className="p-3">{c.hand}</td>
              <td className="p-3">{pct(c.precision)}</td><td className="p-3">{pct(c.recall)}</td>
              <td className="p-3">{pct(c.exact_share)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {Object.keys(data.per_model ?? {}).length > 0 && (
        <table className="w-full border-t border-neutral-800 text-sm">
          <thead className="text-left text-xs uppercase text-neutral-400">
            <tr className="border-b border-neutral-800">
              <th className="p-3">Found by model</th><th className="p-3">Kept</th><th className="p-3">Fixed</th>
              <th className="p-3">Removed</th><th className="p-3">Precision</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(data.per_model).map(([m, c]) => (
              <tr key={m} className="border-b border-neutral-800/60 last:border-0">
                <td className="p-3">{MODEL_NAMES[m] ?? m}</td><td className="p-3">{c.ai_kept}</td>
                <td className="p-3">{c.ai_fixed}</td><td className="p-3">{c.ai_removed}</td>
                <td className="p-3">{pct(c.precision)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}

const MODEL_NAMES: Record<string, string> = {
  yolo: 'YOLOv8s Open Images (601 classes)',
  yolo_coco: 'YOLO26n COCO (80 classes)',
  owlv2: 'OWLv2 (your label names)',
  owlv2_lvis: 'OWLv2 LVIS (1,203 classes)',
  owlv2_similar: 'OWLv2 find alike',
  owlv2_text: 'OWLv2 find by name',
}

function Kpi({ title, value, hint }: { title: string; value: string; hint: string }) {
  return (
    <div className={`${card} p-4`}>
      <div className="text-xs text-neutral-400">{title}</div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
      <div className="mt-1 text-xs text-neutral-500">{hint}</div>
    </div>
  )
}
