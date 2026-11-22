'use client'

import { useEffect, useRef, useState } from 'react'
import { Check, Copy } from 'lucide-react'

export function CopyButton({ text, label = 'Copy message' }: { text: string; label?: string }) {
  const [status, setStatus] = useState('')
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => () => clearTimeout(timer.current), [])
  async function copy() {
    clearTimeout(timer.current)
    try {
      await navigator.clipboard.writeText(text)
      setStatus('Copied')
      timer.current = setTimeout(() => setStatus(''), 2000)
    } catch {
      setStatus('Copy failed. Please select and copy the text.')
    }
  }
  return <span className="copy-control">
    <button type="button" className="copy-button" onClick={copy} title={label} aria-label={label}>
      {status === 'Copied' ? <Check size={14} /> : <Copy size={14} />}{status === 'Copied' ? 'Copied' : 'Copy'}
    </button>
    <span className={status.startsWith('Copy failed') ? 'copy-error' : 'sr-only'} role="status">{status}</span>
  </span>
}
