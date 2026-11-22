import { strict as assert } from 'node:assert'
import { test } from 'node:test'
import { citationNumbers, evidenceLabel, groupSources, remarkCitations } from './citations'
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

test('collects single, grouped, whitespace and bold citations without duplicates', () => {
  assert.deepEqual(citationNumbers('[1] [1,2, 3] **[4, 5]** [ 6 ] [2]'), [1, 2, 3, 4, 5, 6])
})

test('does not treat unfinished groups or nonnumeric brackets as citations', () => {
  assert.deepEqual(citationNumbers('[1, text] [2, '), [])
})

test('ignores citations in code, links, images, and math', () => {
  assert.deepEqual(citationNumbers('Real [1]. `[2]`\n\n```py\na = [3]\n```\n\n[[4]](https://example.com) ![[5]](image.png) $[6]$\n\n$$\n[7]\n$$\n\n**[8, 9]**'), [1, 8, 9])
})


test('counts distinct valid citations and known papers, not retrieved passages', () => {
  const sources = [{ id: 1, paperId: 'a' }, { id: 2, paperId: 'a' }, { id: 3, paperId: 'b' }]
  assert.equal(evidenceLabel('[1] [1, 2] [0, 99]', sources), '2 cited passages · 1 paper')
  assert.equal(evidenceLabel('[1, 3]', sources), '2 cited passages · 2 papers')
  assert.equal(evidenceLabel('`[1]` [99]', sources), 'No sources cited')
  assert.equal(evidenceLabel('[1, 2]', [{ id: 1, paperId: 'a' }, { id: 2 }]), '2 cited passages')
})

test('groups by paper in first appearance order and preserves original passage IDs', () => {
  const sources = [{ id: 1, paperId: 'b' }, { id: 2 }, { id: 3, paperId: 'b' }, { id: 4, paperId: 'a' }, { id: 5, paperId: ' ' }]
  assert.deepEqual(groupSources(sources).map((group) => group.map((source) => source.id)), [[1, 3], [2], [4], [5]])
})

test('citation rendering only creates buttons for valid prose references', () => {
  const processor = unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkCitations, { sourceCount: 2 })
  const tree = processor.runSync(processor.parse('[0, 1, 99] **[2]** `[1]` $[2]$ [[1]](https://example.com)'))
  const serialized = JSON.stringify(tree)
  assert.equal((serialized.match(/data-citation/g) || []).length, 2)
  assert.ok(serialized.includes('https://example.com'))
})
