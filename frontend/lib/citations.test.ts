import { strict as assert } from 'node:assert'
import { test } from 'node:test'
import { citationNumbers } from './citations'

test('collects single, grouped, whitespace and bold citations without duplicates', () => {
  assert.deepEqual(citationNumbers('[1] [1,2, 3] **[4, 5]** [ 6 ] [2]'), [1, 2, 3, 4, 5, 6])
})

test('does not treat unfinished groups or nonnumeric brackets as citations', () => {
  assert.deepEqual(citationNumbers('[1, text] [2, '), [])
})
