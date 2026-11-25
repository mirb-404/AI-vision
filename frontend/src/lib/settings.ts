import type { Mode } from '../api/types'

/** Per-project labeling settings remembered in this browser only. */
export interface SessionSettings {
  mode: Mode
  participant: string
  /** Always-on assistant: run the detector automatically on every image opened in AI mode. */
  autoSuggest: boolean
  /** Add the detector's confident boxes straight away; only unsure ones are asked about. */
  autoAccept: boolean
  /** Show the full tool set (polygon, confidence slider, label setup, shortcuts). */
  expert: boolean
}

const DEFAULTS: SessionSettings = { mode: 'ai', participant: '', autoSuggest: true, autoAccept: true, expert: false }

const key = (projectId: number) => `annotator.settings.${projectId}`

export function loadSettings(projectId: number): SessionSettings {
  try {
    const raw = localStorage.getItem(key(projectId))
    if (raw) {
      const parsed = JSON.parse(raw)
      return {
        mode: parsed.mode === 'manual' ? 'manual' : 'ai',
        participant: String(parsed.participant ?? ''),
        autoSuggest: parsed.autoSuggest !== false,
        autoAccept: parsed.autoAccept !== false,
        expert: parsed.expert === true,
      }
    }
  } catch {
    /* storage unavailable or corrupt: use defaults */
  }
  return { ...DEFAULTS }
}

export function saveSettings(projectId: number, settings: SessionSettings) {
  try {
    localStorage.setItem(key(projectId), JSON.stringify(settings))
  } catch {
    /* ignore */
  }
}

/** Change one setting and keep the rest. */
export function saveSetting<K extends keyof SessionSettings>(projectId: number, k: K, v: SessionSettings[K]) {
  saveSettings(projectId, { ...loadSettings(projectId), [k]: v })
}
