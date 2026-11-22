'use client'

import { useLayoutEffect, useRef, useState } from 'react'
import { BookOpen, X } from 'lucide-react'
import type { Source } from '../app/chat-page'
import { citationNumbers, groupSources } from '../lib/citations'
import { SourceImages } from './source-images'

function Passage({ source, cited, highlighted, conversationId }: { source: Source; cited: boolean; highlighted: boolean; conversationId: string }) {
  const [expanded, setExpanded] = useState(false)
  const [long, setLong] = useState(false)
  const excerptRef = useRef<HTMLParagraphElement>(null)
  useLayoutEffect(() => {
    if (highlighted) setExpanded(true)
  }, [highlighted])
  useLayoutEffect(() => {
    const excerpt = excerptRef.current
    if (!excerpt || expanded) return
    const observer = new ResizeObserver(() => setLong(excerpt.scrollHeight > excerpt.clientHeight + 1))
    observer.observe(excerpt)
    return () => observer.disconnect()
  }, [expanded, source.excerpt])
  const text = source.excerpt.replace(/\[([^\]]+)\]\(#bib\.[^)]+\)/g, '$1').replace(/\s+([,.;:!?])/g, '$1').replace(/\(\s+/g, '(').replace(/\s+\)/g, ')')
  return <section id={`source-${source.id}`} className={`source-passage ${highlighted ? 'highlighted' : ''}`}>
    <div className="passage-heading"><span className="citation-badge">[{source.id}]</span><span>{cited ? 'Cited' : 'Retrieved, not cited'}</span></div>
    {text && <><p id={`passage-${source.id}`} ref={excerptRef} className={`passage-excerpt ${expanded ? '' : 'clamped'}`}>{text}</p>{(long || expanded) && <button className="passage-toggle" type="button" aria-expanded={expanded} aria-controls={`passage-${source.id}`} onClick={() => setExpanded(!expanded)}>{expanded ? 'Show less' : 'Show passage'}</button>}</>}
    <SourceImages conversationId={conversationId} images={source.images.map((image) => ({ ...image, sources: [source.id] }))} />
  </section>
}

export function SourcesPanel({ conversationId, activeSources, content, answerNumber, onClose, highlightedSource }: { conversationId: string; activeSources: Source[]; content: string; answerNumber: number | null; onClose: () => void; highlightedSource: number | null }) {
  const cited = new Set(citationNumbers(content))
  return <aside className="sources-panel flex h-full w-[312px] shrink-0 flex-col border-l border-border bg-card">
    <div className="flex items-center justify-between border-b border-border px-5 py-4"><div><div className="flex items-center gap-2"><BookOpen size={16} className="text-primary" /><h2 className="text-sm font-semibold">Sources</h2></div><p className="mt-1 text-xs text-muted-foreground">{answerNumber ? `Answer ${answerNumber} · ` : ''}{activeSources.length} retrieved passages</p></div><button onClick={onClose} className="icon-button" aria-label="Close sources"><X size={17} /></button></div>
    <div className="flex-1 overflow-auto px-4 py-4">{groupSources(activeSources).map((group) => {
      const source = group[0]
      return <article key={source.id} className="source-card mb-3 rounded-xl border border-border bg-background p-3.5">
        <h3 className="text-[13px] font-semibold leading-5">{source.title}</h3>
        {source.paperId && <p className="mt-1 text-[10px] font-mono text-muted-foreground">arXiv:{source.paperId}</p>}
        <p className="mt-2 text-[11px] leading-4 text-muted-foreground">{source.authors}</p>
        <div className="mt-3 flex items-center justify-between text-[11px] text-muted-foreground"><span>{source.year || 'Retrieved source'}</span>{source.paperId && <a className="font-medium text-primary" href={`https://arxiv.org/abs/${source.paperId}`} target="_blank" rel="noreferrer">View paper ↗</a>}</div>
        {group.map((passage) => <Passage key={passage.id} source={passage} cited={cited.has(passage.id)} highlighted={highlightedSource === passage.id} conversationId={conversationId} />)}
      </article>
    })}{!activeSources.length && <p className="px-1 py-3 text-sm text-muted-foreground">Sources will appear when the assistant retrieves evidence.</p>}</div>
  </aside>
}
