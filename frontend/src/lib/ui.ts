// Shared Tailwind class strings.
export const btn =
  'inline-flex items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition ' +
  'disabled:cursor-not-allowed disabled:opacity-40'
export const btnPrimary = `${btn} bg-indigo-600 text-white hover:bg-indigo-500`
export const btnSecondary = `${btn} bg-slate-800 text-slate-100 hover:bg-slate-700 border border-slate-700`
export const btnDanger = `${btn} bg-transparent text-red-400 hover:bg-red-500/10`
export const input =
  'rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-sm text-slate-100 ' +
  'placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none'
export const card = 'rounded-lg border border-slate-800 bg-slate-900/60'

export const statusStyle: Record<string, string> = {
  new: 'bg-slate-700 text-slate-200',
  in_progress: 'bg-amber-500/20 text-amber-300',
  done: 'bg-emerald-500/20 text-emerald-300',
}

export const fmt = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v) ? '–' : v.toFixed(digits)
