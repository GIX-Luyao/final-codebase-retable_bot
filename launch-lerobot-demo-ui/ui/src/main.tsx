import { StrictMode, useState, useEffect } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import PreflightCheck from './PreflightCheck.tsx'

/**
 * Root component that routes between:
 * - PreflightCheck: when URL has ?preflight or server is in preflight mode
 * - App: the main robot control UI
 *
 * URL modes:
 *   http://localhost:5173/?preflight   → preflight camera setup
 *   http://localhost:5173/             → main control UI
 */
function Root() {
  const [mode, setMode] = useState<'loading' | 'preflight' | 'control'>('loading')

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)

    // Check URL param first
    if (params.has('preflight')) {
      setMode('preflight')
      return
    }

    // Auto-detect: check if preflight server is running
    fetch('/api/preflight/health')
      .then(r => r.json())
      .then(d => {
        if (d.status === 'ok' && d.service === 'preflight') {
          setMode('preflight')
        } else {
          setMode('control')
        }
      })
      .catch(() => {
        // Preflight server not running, go to control mode
        setMode('control')
      })
  }, [])

  const handlePreflightComplete = () => {
    // Remove ?preflight from URL and switch to control mode
    const url = new URL(window.location.href)
    url.searchParams.delete('preflight')
    window.history.replaceState({}, '', url.toString())
    setMode('control')
  }

  if (mode === 'loading') {
    return (
      <div className="w-full min-h-screen bg-gradient-to-b from-gray-950 via-gray-900 to-gray-950 text-white flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-gray-700 border-t-cyan-400 rounded-full animate-spin" />
          <p className="text-gray-400 text-lg">Detecting mode...</p>
        </div>
      </div>
    )
  }

  if (mode === 'preflight') {
    return <PreflightCheck onComplete={handlePreflightComplete} />
  }

  return <App />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
)
