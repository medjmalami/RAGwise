'use client'

import { FormEvent, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { ArrowUp, BookOpen, Check, ChevronDown, LogOut, Menu, MessageSquare, Moon, MoreHorizontal, PanelRight, Search, Settings2, Sparkles, Sun, X } from 'lucide-react'
import { ApiSource, ApiUser, getConversation, getConversations, restoreSession, signIn, signOut, signUp, streamAnswer } from '../lib/api'

type Source = { id: number; title: string; authors: string; year: string; excerpt: string; tag: string; paperId?: string }
type Message = { role: 'user' | 'assistant'; content: string; sources?: Source[]; loading?: boolean }
type Conversation = { id: string; title: string; time: string; messages: Message[]; messagesLoaded: boolean; hasMoreMessages?: boolean; nextBeforeId?: number | null }

function sourceFromApi(source: ApiSource, index: number): Source {
  const payload = source.payload || {}
  const text = source.text || String(payload.text || '')
  const paperId = String(source.paper_id || payload.paper_id || '')
  return {
    id: index + 1,
    title: String(source.title || payload.title || payload.paper_title || (paperId ? `Paper ${paperId}` : `Retrieved passage ${index + 1}`)),
    authors: String(source.authors?.join(', ') || payload.authors || payload.author || 'Research paper'),
    year: String(source.year || payload.year || payload.published || ''),
    excerpt: text,
    tag: String(payload.category || (paperId ? 'Research paper' : 'Retrieved passage')),
    paperId,
  }
}

function cleanPassageText(text: string): string {
  return text
    .replace(/\[([^\]]+)\]\(#bib\.[^)]+\)/g, '$1')
    .replace(/\s+([,.;:!?])/g, '$1')
    .replace(/\(\s+/g, '(')
    .replace(/\s+\)/g, ')')
}

function Logo({ compact = false }: { compact?: boolean }) {
  return <div className="flex items-center gap-2.5"><div className="logo-mark"><span /><span /><span /></div>{!compact && <span className="text-[15px] font-semibold tracking-[-0.02em]">RAGWise</span>}</div>
}

function Sidebar({ conversations, activeId, onSelect, onNew, onClose, isDark, onToggleTheme, onToggleCollapse, sidebarCollapsed, user, onSignOut, hasMoreConversations, isLoadingConversations, onLoadMoreConversations }: { conversations: Conversation[]; activeId: string | null; onSelect: (id: string) => void; onNew: () => void; onClose?: () => void; isDark: boolean; onToggleTheme: () => void; onToggleCollapse?: () => void; sidebarCollapsed?: boolean; user: ApiUser; onSignOut: () => void; hasMoreConversations: boolean; isLoadingConversations: boolean; onLoadMoreConversations: () => void }) {
  return <aside className={`sidebar-panel flex h-full w-[274px] shrink-0 flex-col border-r border-border bg-sidebar px-3 py-4 ${sidebarCollapsed ? 'is-collapsed' : ''}`}>
    <div className="mb-8 flex items-center justify-between px-2"><Logo /><div className="flex items-center gap-1"><button onClick={onToggleCollapse} className="icon-button desktop-only-control" aria-label={sidebarCollapsed ? 'Expand navigation' : 'Collapse navigation'}><Menu size={18} /></button><button onClick={onClose} className="icon-button mobile-only-control" aria-label="Close navigation"><X size={18} /></button></div></div>
    <button className="new-chat mb-7 flex items-center gap-2.5 rounded-lg px-3.5 py-2.5 text-sm font-medium" onClick={onNew}><span className="new-chat-icon" aria-hidden="true">+</span><span>New chat</span></button>
    <div className="mb-2 flex items-center justify-between px-2 text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground"><span>Recent chats</span><Search size={14} /></div>
    <nav className="flex flex-1 flex-col gap-0.5 overflow-auto">{conversations.length ? conversations.map((conversation) => <button key={conversation.id} onClick={() => { onSelect(conversation.id); onClose?.() }} className={`history-item flex items-center gap-2 rounded-lg px-2.5 py-2.5 text-left text-[13px] ${activeId === conversation.id ? 'active' : ''}`}><MessageSquare size={15} className="shrink-0 text-muted-foreground" /><span className="truncate">{conversation.title}</span></button>) : <p className="px-2.5 py-3 text-xs text-muted-foreground">Your conversations will appear here.</p>}</nav>
    {hasMoreConversations && <button type="button" onClick={onLoadMoreConversations} disabled={isLoadingConversations} className="mt-2 rounded-lg px-2.5 py-2 text-left text-xs text-muted-foreground hover:bg-muted disabled:opacity-50">{isLoadingConversations ? 'Loading…' : 'Load older conversations'}</button>}
    <div className="sidebar-footer mt-5 border-t border-border pt-3"><button className="sidebar-link" onClick={onSignOut}><LogOut size={16} />Sign out</button><button className="sidebar-link" onClick={onToggleTheme} aria-label={`Switch to ${isDark ? 'light' : 'dark'} theme`}><div className="theme-icon">{isDark ? <Moon size={14} /> : <Sun size={14} />}</div>Theme <span className="theme-name">{isDark ? 'Dark' : 'Light'}</span><ChevronDown size={14} className="ml-auto text-muted-foreground" /></button><div className="mt-4 flex items-center gap-2.5 rounded-lg px-2.5 py-2"><div className="avatar">{user.name.trim().split(/\s+/).map((part) => part[0]).join('').slice(0, 2).toUpperCase()}</div><div className="min-w-0"><p className="truncate text-xs font-medium">{user.name}</p><p className="truncate text-[11px] text-muted-foreground">{user.email}</p></div><MoreHorizontal size={16} className="ml-auto text-muted-foreground" /></div></div>
  </aside>
}

function SourcesPanel({ activeSources, answerNumber, onClose, highlightedSource }: { activeSources: Source[]; answerNumber: number | null; onClose: () => void; highlightedSource: number | null }) {
  return <aside className="sources-panel flex h-full w-[312px] shrink-0 flex-col border-l border-border bg-card"><div className="flex items-center justify-between border-b border-border px-5 py-4"><div><div className="flex items-center gap-2"><BookOpen size={16} className="text-primary" /><h2 className="text-sm font-semibold">Sources</h2></div><p className="mt-1 text-xs text-muted-foreground">{answerNumber ? `Answer ${answerNumber} · ` : ''}{activeSources.length} retrieved passages</p></div><button onClick={onClose} className="icon-button" aria-label="Close sources"><X size={17} /></button></div><div className="flex-1 overflow-auto px-4 py-4">{activeSources.length ? activeSources.map((source, index) => <article id={`source-${index + 1}`} key={`${source.paperId}-${source.id}`} className={`source-card group mb-3 rounded-xl border border-border bg-background p-3.5 ${highlightedSource === index + 1 ? 'highlighted' : ''}`}><div className="mb-3 flex items-start justify-between"><span className="citation-badge">[{index + 1}]</span><span className="rounded-full bg-muted px-2 py-1 text-[10px] font-medium text-muted-foreground">{source.tag}</span></div><h3 className="text-[13px] font-semibold leading-5 text-foreground">{source.title}</h3>{source.paperId && <p className="mt-1 text-[10px] font-mono text-muted-foreground">arXiv:{source.paperId}</p>}{source.authors && <p className="mt-2 text-[11px] leading-4 text-muted-foreground">{source.authors}</p>}<div className="mt-3 flex items-center justify-between border-t border-border pt-3 text-[11px] text-muted-foreground"><span>{source.year || 'Retrieved source'}</span>{source.paperId && <a className="font-medium text-primary" href={`https://arxiv.org/abs/${source.paperId}`} target="_blank" rel="noreferrer">View paper ↗</a>}</div>{source.excerpt && <p className="mt-3 border-l-2 border-primary/30 pl-3 text-xs leading-5 text-muted-foreground">{cleanPassageText(source.excerpt)}</p>}</article>) : <p className="px-1 py-3 text-sm text-muted-foreground">Sources will appear when the assistant retrieves evidence.</p>}</div></aside>
}

function renderInline(text: string, onCitation: (index: number) => void, sourceCount: number) {
  return text.replace(/\$\s*\\(?:rightarrow|to|Rightarrow)\s*\$/g, '→').split(/(\[\d+\]|\*\*[^*]+\*\*)/g).map((part, index) => {
    const citation = part.match(/^\[(\d+)\]$/)
    if (citation) {
      const sourceIndex = Number(citation[1])
      return sourceIndex > 0 && sourceIndex <= sourceCount
        ? <button key={index} type="button" className="citation-link" onClick={() => onCitation(sourceIndex)}>{part}</button>
        : <span key={index} className="citation-link">{part}</span>
    }
    return part.startsWith('**') ? <strong key={index}>{part.slice(2, -2)}</strong> : part
  })
}

function MarkdownText({ content, onCitation, sourceCount }: { content: string; onCitation: (index: number) => void; sourceCount: number }) {
  const normalized = content
    .replace(/(^|\n)\s*\*\s+/g, '$1- ')
    .replace(/([:.;])\s+\*\s+(?=\S)/g, '$1\n- ')
  const blocks: Array<{ type: 'paragraph'; text: string } | { type: 'list'; items: string[] }> = []
  let paragraph: string[] = []
  let items: string[] = []
  const flushParagraph = () => {
    if (paragraph.length) blocks.push({ type: 'paragraph', text: paragraph.join(' ') })
    paragraph = []
  }
  const flushItems = () => {
    if (items.length) blocks.push({ type: 'list', items })
    items = []
  }

  for (const line of normalized.split('\n')) {
    const trimmed = line.trim()
    const bullet = line.match(/^\s*[-*+]\s+(.+)$/)
    if (!trimmed) {
      flushParagraph()
      flushItems()
    } else if (bullet) {
      flushParagraph()
      items.push(bullet[1])
    } else {
      flushItems()
      paragraph.push(trimmed)
    }
  }
  flushParagraph()
  flushItems()

  return <div className="message-copy">{blocks.map((block, index) => block.type === 'list'
    ? <ul key={index}>{block.items.map((item, itemIndex) => <li key={itemIndex}>{renderInline(item, onCitation, sourceCount)}</li>)}</ul>
    : <p key={index}>{renderInline(block.text, onCitation, sourceCount)}</p>)}</div>
}

function AuthScreen({ onAuthenticated }: { onAuthenticated: (user: ApiUser) => void }) {
  const [mode, setMode] = useState<'signin' | 'signup'>('signin')
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  async function submit(event: FormEvent) {
    event.preventDefault(); setError(''); setBusy(true)
    try { onAuthenticated(mode === 'signup' ? await signUp(name, email, password) : await signIn(email, password)) }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Unable to authenticate') }
    finally { setBusy(false) }
  }
  return <main className="auth-screen"><form className="auth-card" onSubmit={submit}><Logo /><p className="eyebrow">Your research copilot</p><h1>{mode === 'signin' ? 'Welcome back' : 'Create your account'}</h1><p className="auth-subtitle">Sign in to ask questions and explore evidence from the RAGWise collection.</p>{mode === 'signup' && <label>Name<input autoComplete="name" required minLength={1} maxLength={255} value={name} onChange={(event) => setName(event.target.value)} /></label>}<label>Email<input type="email" autoComplete="email" required value={email} onChange={(event) => setEmail(event.target.value)} /></label><label>Password<input type="password" autoComplete={mode === 'signup' ? 'new-password' : 'current-password'} required minLength={mode === 'signup' ? 8 : 1} maxLength={128} value={password} onChange={(event) => setPassword(event.target.value)} /></label>{error && <p className="auth-error" role="alert">{error}</p>}<button className="auth-submit" type="submit" disabled={busy}>{busy ? 'Please wait…' : mode === 'signup' ? 'Create account' : 'Sign in'}</button><p className="auth-switch">{mode === 'signin' ? 'New to RAGWise?' : 'Already have an account?'} <button type="button" onClick={() => { setMode(mode === 'signin' ? 'signup' : 'signin'); setError('') }}>{mode === 'signin' ? 'Create an account' : 'Sign in'}</button></p></form></main>
}

export default function Page() {
  const [user, setUser] = useState<ApiUser | null>(null)
  const [authLoading, setAuthLoading] = useState(true)
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [hasMoreConversations, setHasMoreConversations] = useState(false)
  const [isLoadingConversations, setIsLoadingConversations] = useState(false)
  const [conversationCursor, setConversationCursor] = useState<{ updatedAt: string; id: string } | null>(null)
  const [activeId, setActiveId] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [showSources, setShowSources] = useState(false)
  const [highlightedSource, setHighlightedSource] = useState<number | null>(null)
  const [sourceMessageIndex, setSourceMessageIndex] = useState<number | null>(null)
  const [showSidebar, setShowSidebar] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [isDark, setIsDark] = useState(true)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isLoadingOlder, setIsLoadingOlder] = useState(false)
  const [error, setError] = useState('')
  const chatScrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => { restoreSession().then(setUser).finally(() => setAuthLoading(false)) }, [])
  useEffect(() => {
    if (!user) { setConversations([]); setActiveId(null); setHasMoreConversations(false); setConversationCursor(null); return }
    let cancelled = false
    getConversations().then((page) => {
      if (cancelled) return
      const mapped = page.conversations.map((item) => ({
        id: item.id,
        title: item.title || 'Untitled conversation',
        time: item.updated_at,
        messages: [],
        messagesLoaded: false,
        hasMoreMessages: false,
        nextBeforeId: null,
      }))
      setConversations(mapped)
      setHasMoreConversations(page.has_more)
      setConversationCursor(page.next_before_updated_at && page.next_before_id ? { updatedAt: page.next_before_updated_at, id: page.next_before_id } : null)
      setActiveId(mapped[0]?.id ?? null)
    }).catch((cause) => {
      if (!cancelled) setError(cause instanceof Error ? cause.message : 'Unable to load conversations')
    })
    return () => { cancelled = true }
  }, [user])
  async function loadMoreConversations() {
    if (!hasMoreConversations || !conversationCursor || isLoadingConversations) return
    setIsLoadingConversations(true)
    try {
      const page = await getConversations(conversationCursor.updatedAt, conversationCursor.id)
      const mapped = page.conversations.map((item) => ({
        id: item.id,
        title: item.title || 'Untitled conversation',
        time: item.updated_at,
        messages: [],
        messagesLoaded: false,
        hasMoreMessages: false,
        nextBeforeId: null,
      }))
      setConversations((items) => [...items, ...mapped.filter((item) => !items.some((existing) => existing.id === item.id))])
      setHasMoreConversations(page.has_more)
      setConversationCursor(page.next_before_updated_at && page.next_before_id ? { updatedAt: page.next_before_updated_at, id: page.next_before_id } : null)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to load more conversations')
    } finally { setIsLoadingConversations(false) }
  }
  useEffect(() => {
    if (!activeId || activeId.startsWith('pending-')) return
    const conversation = conversations.find((item) => item.id === activeId)
    if (!conversation || conversation.messagesLoaded) return
    let cancelled = false
    getConversation(activeId).then((detail) => {
      if (cancelled) return
      const messages: Message[] = detail.messages.map((message) => ({
        role: message.role,
        content: message.content,
        sources: (message.sources || []).map((source, index) => sourceFromApi(source, index)),
      }))
      setConversations((items) => items.map((item) => item.id === detail.id
        ? { ...item, title: detail.title || item.title, time: detail.updated_at, messages, messagesLoaded: true, hasMoreMessages: detail.has_more, nextBeforeId: detail.next_before_id }
        : item))
    }).catch((cause) => {
      if (!cancelled) setError(cause instanceof Error ? cause.message : 'Unable to load conversation')
    })
    return () => { cancelled = true }
  }, [activeId, conversations])

  async function loadOlderMessages() {
    if (!active || !active.hasMoreMessages || active.nextBeforeId == null || isLoadingOlder) return
    setIsLoadingOlder(true)
    try {
      const detail = await getConversation(active.id, active.nextBeforeId)
      const olderMessages: Message[] = detail.messages.map((message) => ({
        role: message.role,
        content: message.content,
        sources: (message.sources || []).map((source, index) => sourceFromApi(source, index)),
      }))
      setConversations((items) => items.map((item) => item.id === detail.id
        ? { ...item, messages: [...olderMessages, ...item.messages], hasMoreMessages: detail.has_more, nextBeforeId: detail.next_before_id }
        : item))
      setSourceMessageIndex((index) => index === null ? null : index + olderMessages.length)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to load earlier messages')
    } finally { setIsLoadingOlder(false) }
  }

  const active = conversations.find((conversation) => conversation.id === activeId)
  const isLoadingActive = Boolean(activeId && active && !active.messagesLoaded)
  const latestAssistantIndex = active?.messages.reduce((last, message, index) => message.role === 'assistant' ? index : last, -1) ?? -1
  const displayedSourceMessageIndex = sourceMessageIndex !== null && active?.messages[sourceMessageIndex]?.role === 'assistant'
    ? sourceMessageIndex
    : latestAssistantIndex
  const displayedAnswerNumber = displayedSourceMessageIndex >= 0
    ? active?.messages.slice(0, displayedSourceMessageIndex + 1).filter((message) => message.role === 'assistant').length ?? null
    : null
  const activeSources = displayedSourceMessageIndex >= 0 ? active?.messages[displayedSourceMessageIndex]?.sources ?? [] : []
  function openSource(messageIndex: number, sourceIndex: number) {
    setSourceMessageIndex(messageIndex)
    setShowSources(true)
    setHighlightedSource(sourceIndex)
    window.requestAnimationFrame(() => document.getElementById(`source-${sourceIndex}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }))
    window.setTimeout(() => setHighlightedSource((current) => current === sourceIndex ? null : current), 1800)
  }
  useLayoutEffect(() => {
    if (!active?.messagesLoaded) return
    const chatScroll = chatScrollRef.current
    if (chatScroll) chatScroll.scrollTop = chatScroll.scrollHeight
  }, [activeId, active?.messagesLoaded])
  const newChat = () => { if (isStreaming) return; setActiveId(null); setSourceMessageIndex(null); setInput(''); setError(''); setShowSources(false); setShowSidebar(false) }

  const send = useCallback(async (text = input) => {
    const query = text.trim()
    if (!query || isStreaming || isLoadingActive || !user) return
    setInput(''); setError(''); setIsStreaming(true)
    const previous = conversations.find((conversation) => conversation.id === activeId)
    setSourceMessageIndex(null)
    const temporaryId = previous?.id || `pending-${Date.now()}`
    const userMessage: Message = { role: 'user', content: query }
    const placeholder: Message = { role: 'assistant', content: '', sources: [], loading: true }
    const initial: Conversation = previous ? { ...previous, messages: [...previous.messages, userMessage, placeholder], messagesLoaded: true } : { id: temporaryId, title: query.slice(0, 60), time: 'Just now', messages: [userMessage, placeholder], messagesLoaded: true, hasMoreMessages: false, nextBeforeId: null }
    setConversations((items) => previous ? items.map((item) => item.id === temporaryId ? initial : item) : [initial, ...items])
    setActiveId(temporaryId)
    let committedId = temporaryId
    try {
      await streamAnswer(query, previous?.id || null, (event, data) => {
        if (event === 'title' && data.title) {
          const targetId = committedId
          setConversations((items) => items.map((item) => item.id === targetId ? { ...item, title: data.title } : item))
        }
        if (event === 'sources') {
          const targetId = committedId
          const sources = (Array.isArray(data) ? data : []).map((source: ApiSource, index: number) => sourceFromApi(source, index))
          setConversations((items) => items.map((item) => item.id === targetId ? { ...item, messages: item.messages.map((message, index) => index === item.messages.length - 1 && message.role === 'assistant' ? { ...message, sources } : message) } : item))
        }
        if (event === 'token') {
          const targetId = committedId
          setConversations((items) => items.map((item) => item.id === targetId ? { ...item, messages: item.messages.map((message, index) => index === item.messages.length - 1 && message.role === 'assistant' ? { ...message, content: message.content + (data.text || '') } : message) } : item))
        }
        if (event === 'reset') {
          const targetId = committedId
          setConversations((items) => items.map((item) => item.id === targetId ? { ...item, messages: item.messages.map((message, index) => index === item.messages.length - 1 && message.role === 'assistant' ? { ...message, content: '' } : message) } : item))
        }
        if (event === 'done') {
          const serverId = String(data.conversation_id)
          const previousId = committedId
          const wasPending = previousId.startsWith('pending-')
          const finalId = serverId
          setConversations((items) => items.map((item) => item.id === previousId ? { ...item, id: finalId, title: data.title || item.title, messages: item.messages.map((message, index) => index === item.messages.length - 1 && message.role === 'assistant' ? { ...message, content: data.answer || message.content, loading: false } : message) } : item))
          committedId = finalId
          if (wasPending) setActiveId(finalId)
        }
        if (event === 'error') throw new Error(data.detail || 'The answer could not be generated')
      })
      setConversations((items) => items.map((item) => item.id === committedId ? { ...item, messages: item.messages.map((message) => message.loading ? { ...message, loading: false } : message) } : item))
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : 'Unable to reach the RAGWise API'
      setError(message)
      if (committedId.startsWith('pending-')) {
        setConversations((items) => items.filter((item) => item.id !== committedId))
        setActiveId(null)
      } else {
        setConversations((items) => items.map((item) => item.id === committedId ? { ...item, messages: item.messages.filter((entry) => !entry.loading || entry.content) } : item))
      }
    } finally { setIsStreaming(false) }
  }, [activeId, conversations, input, isLoadingActive, isStreaming, user])

  async function logout() { try { await signOut() } catch { /* Clear the local session even when the API is unavailable. */ } setUser(null) }

  if (authLoading) return <div className="auth-loading"><Logo /><span>Connecting to RAGWise…</span></div>
  if (!user) return <div className={isDark ? 'dark' : ''}><AuthScreen onAuthenticated={setUser} /></div>


  return <div className={isDark ? 'dark app-shell' : 'app-shell'}><div className="app-frame"><div className={`mobile-overlay ${showSidebar || showSources ? 'visible' : ''}`} onClick={() => { setShowSidebar(false); setShowSources(false) }} /><div className={`sidebar-drawer ${showSidebar ? 'open' : ''}`}><Sidebar conversations={conversations} activeId={activeId} onSelect={(id) => { if (!isStreaming) { setSourceMessageIndex(null); setActiveId(id) } }} onNew={newChat} onClose={() => setShowSidebar(false)} isDark={isDark} onToggleTheme={() => setIsDark((dark) => !dark)} onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)} sidebarCollapsed={sidebarCollapsed} user={user} onSignOut={logout} hasMoreConversations={hasMoreConversations} isLoadingConversations={isLoadingConversations} onLoadMoreConversations={loadMoreConversations} /></div><div className={`desktop-sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}><Sidebar conversations={conversations} activeId={activeId} onSelect={(id) => { if (!isStreaming) { setSourceMessageIndex(null); setActiveId(id) } }} onNew={newChat} isDark={isDark} onToggleTheme={() => setIsDark((dark) => !dark)} onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)} sidebarCollapsed={sidebarCollapsed} user={user} onSignOut={logout} hasMoreConversations={hasMoreConversations} isLoadingConversations={isLoadingConversations} onLoadMoreConversations={loadMoreConversations} /></div><main className="chat-area flex min-w-0 flex-1 flex-col"><header className="chat-header flex items-center justify-between border-b border-border px-4 py-3 md:px-7"><button className="icon-button mobile-only-control" onClick={() => setShowSidebar(true)} aria-label="Open navigation"><Menu size={19} /></button><div className="ml-auto flex items-center gap-1.5"><button className={`sources-toggle flex items-center gap-2 rounded-lg px-2.5 py-2 text-xs font-medium ${showSources ? 'selected' : ''}`} onClick={() => { if (!showSources) setSourceMessageIndex(latestAssistantIndex); setShowSources(!showSources) }}><PanelRight size={16} /> <span className="hidden sm:inline">Sources</span></button></div></header><div ref={chatScrollRef} className="chat-scroll flex-1 overflow-auto"><div className={`chat-content mx-auto w-full ${active ? 'has-messages' : ''}`}>{!active ? <Welcome onPrompt={(prompt) => send(prompt)} /> : isLoadingActive ? <p className="text-center text-sm text-muted-foreground">Loading conversation…</p> : <div className="message-list">{active.hasMoreMessages && <button type="button" className="mx-auto mb-4 block rounded-lg border border-border px-4 py-2 text-xs text-muted-foreground hover:bg-muted disabled:opacity-50" onClick={loadOlderMessages} disabled={isLoadingOlder}>{isLoadingOlder ? 'Loading earlier messages…' : 'Load earlier messages'}</button>}{active.messages.map((message, index) => <div key={index} className={`message-row ${message.role}`}>{message.role === 'assistant' && <div className="assistant-mark"><Sparkles size={15} /></div>}<div className={message.role === 'user' ? 'user-bubble' : 'assistant-bubble'}>{message.role === 'assistant' ? <>{message.content ? <MarkdownText content={message.content} onCitation={(sourceIndex) => openSource(index, sourceIndex)} sourceCount={message.sources?.length ?? 0} /> : message.loading ? <p className="streaming-indicator">Searching papers and drafting an answer…</p> : null}{message.loading && message.content && <span className="streaming-cursor" />}{!message.loading && message.content && <div className="response-footer"><span><Check size={13} /> Answer grounded in {message.sources?.length ?? 0} sources</span><button type="button" className="sources-action rounded-md px-2 py-1 hover:bg-muted disabled:opacity-40" onClick={() => { if (showSources && sourceMessageIndex === index) setShowSources(false); else { setSourceMessageIndex(index); setShowSources(true) } }} disabled={!message.sources?.length} aria-label="Toggle sources for this answer"><BookOpen size={14} /> Sources</button><button aria-label="More answer options"><MoreHorizontal size={15} /></button></div>}</> : <p>{message.content}</p>}</div></div>)}</div>}</div>{error && <div className="chat-error" role="alert">{error}</div>}</div><div className="composer-wrap mx-auto w-full"><div className="composer"><textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing && event.keyCode !== 229) { event.preventDefault(); send() } }} placeholder="Ask a question about RAG research..." rows={1} aria-label="Ask a research question" /><div className="composer-bottom"><div className="flex items-center gap-1.5 text-[11px] text-muted-foreground"><span className="composer-tool"><Sparkles size={14} /> Research mode</span><span className="hidden sm:inline text-muted-foreground/40">•</span><span className="hidden sm:inline">RAGWise collection</span></div><button onClick={() => send()} className="send-button" disabled={!input.trim() || isStreaming || isLoadingActive} aria-label="Send message"><ArrowUp size={17} /></button></div></div><p className="disclaimer">RAGWise can make mistakes. Verify important findings with the original papers.</p></div></main><div className={`sources-drawer ${showSources ? 'open' : ''}`}><SourcesPanel activeSources={activeSources} answerNumber={displayedAnswerNumber} onClose={() => setShowSources(false)} highlightedSource={highlightedSource} /></div></div></div>
}

function Welcome({ onPrompt }: { onPrompt: (text: string) => void }) { const prompts = ['What are the latest approaches to RAG?', 'What are the main RAG evaluation methods?']; return <div className="welcome"><div className="welcome-icon"><Sparkles size={20} /></div><p className="eyebrow">Your research copilot</p><h1>Explore the RAG<br /><span>research landscape.</span></h1><p className="welcome-subtitle">Ask questions, compare methods, and trace every answer back to the papers that support it.</p><div className="prompt-grid">{prompts.map((prompt) => <button key={prompt} onClick={() => onPrompt(prompt)} className="prompt-card"><span>{prompt}</span><ArrowUp size={15} /></button>)}</div></div> }
