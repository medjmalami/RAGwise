import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import type { Root, RootContent } from 'mdast'

const parser = unified().use(remarkParse).use(remarkGfm).use(remarkMath)
const excluded = new Set(['code', 'inlineCode', 'math', 'inlineMath', 'link', 'linkReference', 'image', 'imageReference', 'html'])
const pattern = /\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]/g

function visitText(node: Root | RootContent, visit: (text: string) => RootContent[] | void) {
  if (excluded.has(node.type) || !('children' in node)) return
  node.children = node.children.flatMap((child) => {
    if (child.type === 'text') return visit(child.value) || [child]
    visitText(child, visit)
    return [child]
  }) as typeof node.children
}

export function citationNumbers(text: string): number[] {
  const numbers = new Set<number>()
  visitText(parser.parse(text), (value) => {
    for (const match of value.matchAll(pattern)) for (const number of match[1].split(',').map(Number)) numbers.add(number)
  })
  return [...numbers]
}

export function remarkCitations({ sourceCount }: { sourceCount: number }) {
  return (tree: Root) => visitText(tree, (value) => {
    const nodes: RootContent[] = []
    let offset = 0
    for (const match of value.matchAll(pattern)) {
      nodes.push({ type: 'text', value: value.slice(offset, match.index) + '[' })
      match[1].split(',').map(Number).forEach((number, index) => {
        if (index) nodes.push({ type: 'text', value: ', ' })
        nodes.push(number > 0 && number <= sourceCount
          ? { type: 'link', url: `#source-${number}`, data: { hProperties: { 'data-citation': number } }, children: [{ type: 'text', value: String(number) }] }
          : { type: 'text', value: String(number) })
      })
      nodes.push({ type: 'text', value: ']' })
      offset = match.index! + match[0].length
    }
    nodes.push({ type: 'text', value: value.slice(offset) })
    return nodes
  })
}

type EvidenceSource = { id: number; paperId?: string }
export function citedSources<T extends EvidenceSource>(content: string, sources: T[]): T[] {
  const numbers = new Set(citationNumbers(content))
  return sources.filter((source) => numbers.has(source.id))
}

export function evidenceLabel(content: string, sources: EvidenceSource[]): string {
  const cited = citedSources(content, sources)
  const count = new Set(cited.map((source) => source.id)).size
  if (!count) return 'No sources cited'
  const papers = new Set(cited.map((source) => source.paperId?.trim()).filter(Boolean)).size
  return `${count} cited passage${count === 1 ? '' : 's'}${cited.every((source) => source.paperId?.trim()) ? ` · ${papers} paper${papers === 1 ? '' : 's'}` : ''}`
}

export function groupSources<T extends EvidenceSource>(sources: T[]): T[][] {
  const groups = new Map<string, T[]>()
  for (const source of sources) {
    const key = source.paperId?.trim() ? `paper:${source.paperId.trim()}` : `passage:${source.id}`
    const group = groups.get(key) || []
    group.push(source)
    groups.set(key, group)
  }
  return [...groups.values()]
}
