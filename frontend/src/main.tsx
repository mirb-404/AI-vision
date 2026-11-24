import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router'
import { App } from './App'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {/* Hash routing keeps the page path fixed, so the app works under any URL prefix. */}
    <HashRouter>
      <App />
    </HashRouter>
  </StrictMode>,
)
