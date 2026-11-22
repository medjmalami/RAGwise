export function citationNumbers(text: string): number[] {
  return [...new Set([...text.matchAll(/\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]/g)].flatMap((match) => match[1].split(',').map(Number)))]
}
