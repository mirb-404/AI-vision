import type { Mode } from '../api/types'

/** Per-project labeling settings remembered in this browser only. */
export interface SessionSettings {
  mode: Mode
  participant: string
}

const key = (projectId: number) => `annotator.settings.${projectId}`

export function loadSettings(projectId: number): SessionSettings {
  try {
    const raw = localStorage.getItem(key(projectId))
    if (raw) {
      const parsed = JSON.parse(raw)
      return { mode: parsed.mode === 'manual' ? 'manual' : 'ai', participant: String(parsed.participant ?? '') }
    }
  } catch {
    /* storage unavailable or corrupt: use defaults */
  }
  return { mode: 'ai', participant: '' }
}

export function saveSettings(projectId: number, settings: SessionSettings) {
  try {
    localStorage.setItem(key(projectId), JSON.stringify(settings))
  } catch {
    /* ignore */
  }
}
