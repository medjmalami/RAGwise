'use client'

import { useEffect, useRef, useState } from 'react'
import { fetchSourceImage, SourceImage } from '../lib/api'

export function SourceImages({ conversationId, images }: { conversationId: string; images: (SourceImage & { sources: number[] })[] }) {
  return <div className="source-images">{images.map((image) => <ImageThumbnail key={`${conversationId}:${image.path}`} conversationId={conversationId} image={image} />)}</div>
}

function ImageThumbnail({ conversationId, image }: { conversationId: string; image: SourceImage & { sources: number[] } }) {
  const [url, setUrl] = useState<string>()
  const [failed, setFailed] = useState(false)
  const dialog = useRef<HTMLDialogElement>(null)
  const label = `Image ${image.n} · Source ${image.sources.map((n) => `[${n}]`).join(', ')}`
  useEffect(() => {
    const controller = new AbortController()
    let objectUrl: string | undefined
    fetchSourceImage(conversationId, image.path, controller.signal).then((blob) => {
      if (controller.signal.aborted) return
      objectUrl = URL.createObjectURL(blob)
      setUrl(objectUrl)
    }).catch(() => { if (!controller.signal.aborted) setFailed(true) })
    return () => { controller.abort(); if (objectUrl) URL.revokeObjectURL(objectUrl) }
  }, [conversationId, image.path])
  return <figure className="source-image">
    {failed ? <p>Image unavailable</p> : url ? <button type="button" onClick={() => dialog.current?.showModal()} aria-label={`Enlarge ${label}`}><img src={url} alt={label} /></button> : <p role="status">Loading image…</p>}
    <figcaption>{label}</figcaption>
    <dialog ref={dialog} className="image-viewer" aria-label={label} onClick={(event) => { if (event.target === event.currentTarget) dialog.current?.close() }}>
      <button type="button" autoFocus onClick={() => dialog.current?.close()} aria-label="Close image">Close ×</button>
      {url && <img src={url} alt={label} />}
      <p>{label}</p>
    </dialog>
  </figure>
}
