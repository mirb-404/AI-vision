// Shared Tailwind class strings. Theme: near-black surfaces, neutral grays, white primary.
export const btn =
  'inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium ' +
  'transition duration-150 ease-out active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 ' +
  'focus-visible:outline-neutral-300 disabled:cursor-not-allowed disabled:opacity-40 disabled:active:scale-100'
export const btnPrimary = `${btn} bg-white text-neutral-950 shadow-sm shadow-white/10 hover:bg-neutral-200`
export const btnSecondary =
  `${btn} border border-neutral-800 bg-neutral-900 text-neutral-100 hover:border-neutral-700 hover:bg-neutral-800`
export const btnDanger = `${btn} bg-transparent text-red-400 hover:bg-red-500/10`
export const input =
  'rounded-lg border border-neutral-800 bg-neutral-950 px-2.5 py-1.5 text-sm text-neutral-100 transition ' +
  'placeholder:text-neutral-600 focus:border-neutral-500 focus:outline-none'
export const card = 'rounded-xl border border-neutral-800/80 bg-neutral-900/50 shadow-lg shadow-black/20'

export const statusStyle: Record<string, string> = {
  new: 'bg-neutral-700 text-neutral-200',
  in_progress: 'bg-amber-500/20 text-amber-300',
  done: 'bg-emerald-500/20 text-emerald-300',
}

export const fmt = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v) ? '–' : v.toFixed(digits)
