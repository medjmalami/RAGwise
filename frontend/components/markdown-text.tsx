'use client'

import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { remarkCitations } from '../lib/citations'
import { CopyButton } from './copy-button'
import type { Element, ElementContent } from 'hast'

function nodeText(node: Element | ElementContent): string {
  return node.type === 'text' ? node.value : 'children' in node ? node.children.map(nodeText).join('') : ''
}

export function MarkdownText({ content, onCitation, sourceCount }: { content: string; onCitation: (index: number) => void; sourceCount: number }) {
  return <div className="message-copy"><Markdown skipHtml
    remarkPlugins={[remarkGfm, remarkMath, [remarkCitations, { sourceCount }]]}
    rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false, trust: false }]]}
    components={{
      a: ({ node, children, href }) => {
        const citation = Number(node?.properties['data-citation'])
        return citation ? <button type="button" className="citation-link" aria-label={`Open source ${citation}`} onClick={() => onCitation(citation)}>{children}</button>
          : <a href={href} target="_blank" rel="noreferrer">{children}</a>
      },
      pre: ({ node, children }) => <div className="code-block"><div className="code-toolbar"><CopyButton text={node ? nodeText(node).replace(/\n$/, '') : ''} label="Copy code" /></div><pre>{children}</pre></div>,
      table: ({ children }) => <div className="table-scroll" tabIndex={0} role="region" aria-label="Answer table"><table>{children}</table></div>,
      img: ({ alt }) => <span>{alt}</span>,
    }}
  >{content}</Markdown></div>
}
