export const API_URL = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/\/$/, '')

export type ApiUser = { id: number; email: string; name: string; created_at: string }
export type AuthResponse = { access_token: string; expires_in: number; user: ApiUser }
export type ApiSource = { score?: number | null; chunk_id?: string | null; paper_id?: string | null; text?: string | null; payload?: Record<string, unknown> }
export type ApiConversation = { id: string; title: string | null; created_at: string; updated_at: string }
export type ApiConversationMessage = { id: number; role: 'user' | 'assistant'; content: string; sources: ApiSource[] | null; created_at: string }
export type ApiConversationDetail = ApiConversation & { messages: ApiConversationMessage[]; has_more: boolean; next_before_id: number | null }

const TOKEN_KEY = 'ragwise.access-token'

export function getToken() {
  return typeof window === 'undefined' ? null : window.localStorage.getItem(TOKEN_KEY)
}

function saveAuth(result: AuthResponse) {
  window.localStorage.setItem(TOKEN_KEY, result.access_token)
  return result.user
}

async function request<T>(path: string, options: RequestInit = {}, token = getToken()): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    credentials: 'include',
    headers: {
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  })
  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try {
      const body = await response.json()
      detail = body.detail || detail
    } catch { /* Keep the HTTP status message. */ }
    throw new Error(detail)
  }
  return (response.status === 204 ? undefined : response.json()) as Promise<T>
}

export async function signIn(email: string, password: string) {
  return saveAuth(await request<AuthResponse>('/auth/signin', { method: 'POST', body: JSON.stringify({ email, password }) }, null))
}

export async function signUp(name: string, email: string, password: string) {
  return saveAuth(await request<AuthResponse>('/auth/signup', { method: 'POST', body: JSON.stringify({ name, email, password }) }, null))
}

export async function restoreSession(): Promise<ApiUser | null> {
  const token = getToken()
  if (token) {
    try { return await request<ApiUser>('/auth/me') } catch { /* Try rotating the refresh cookie. */ }
  }
  try {
    const result = await request<AuthResponse>('/auth/refresh', { method: 'POST' }, null)
    return saveAuth(result)
  } catch {
    window.localStorage.removeItem(TOKEN_KEY)
    return null
  }
}

export async function signOut() {
  try { await request<void>('/auth/logout', { method: 'POST' }) } finally { window.localStorage.removeItem(TOKEN_KEY) }
}

export async function getConversations() {
  return request<ApiConversation[]>('/conversations')
}

export async function getConversation(id: string, beforeId?: number) {
  const params = new URLSearchParams({ limit: '10' })
  if (beforeId !== undefined) params.set('before_id', String(beforeId))
  return request<ApiConversationDetail>(`/conversations/${encodeURIComponent(id)}?${params}`)
}

export async function streamAnswer(query: string, conversationId: string | null, onEvent: (event: string, data: any) => void, signal?: AbortSignal) {
  const response = await fetch(`${API_URL}/answer`, {
    method: 'POST',
    credentials: 'include',
    signal,
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${getToken() || ''}`, Accept: 'text/event-stream' },
    body: JSON.stringify({ query, ...(conversationId ? { conversation_id: conversationId } : {}) }),
  })
  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try { detail = (await response.json()).detail || detail } catch { /* Keep the HTTP status message. */ }
    throw new Error(detail)
  }
  if (!response.body) throw new Error('The server did not provide a response stream')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, '\n')
    let boundary
    while ((boundary = buffer.indexOf('\n\n')) >= 0) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      const lines = frame.split('\n')
      const event = lines.find((line) => line.startsWith('event:'))?.slice(6).trim() || 'message'
      const data = lines.filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trim()).join('\n')
      if (data) onEvent(event, JSON.parse(data))
    }
    if (done) break
  }
}
