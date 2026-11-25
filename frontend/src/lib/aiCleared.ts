/**
 * Images where the user removed the AI's labels. The always-on assistant does not label them
 * again on the next visit ("Look again" still works). Remembered in this browser only.
 */
const key = (projectId: number) => `annotator.aiCleared.${projectId}`

function load(projectId: number): Set<number> {
  try {
    return new Set(JSON.parse(localStorage.getItem(key(projectId)) ?? '[]') as number[])
  } catch {
    return new Set()
  }
}

export function isAiCleared(projectId: number, imageId: number): boolean {
  return load(projectId).has(imageId)
}

export function setAiCleared(projectId: number, imageId: number, cleared: boolean) {
  const ids = load(projectId)
  if (cleared) ids.add(imageId)
  else ids.delete(imageId)
  try {
    localStorage.setItem(key(projectId), JSON.stringify([...ids]))
  } catch {
    /* storage unavailable: the assistant just labels again next time */
  }
}
