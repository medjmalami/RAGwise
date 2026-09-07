'use client'

import { useMemo, useState } from 'react'
import {
  ArrowUp,
  BookOpen,
  Check,
  ChevronDown,
  Command,
  FileText,
  Globe2,
  Menu,
  MessageSquare,
  Moon,
  MoreHorizontal,
  PanelRight,
  Plus,
  Search,
  Settings2,
  Sparkles,
  Sun,
  X,
} from 'lucide-react'

type Source = {
  id: number
  title: string
  authors: string
  year: string
  excerpt: string
  tag: string
}

type Message = { role: 'user' | 'assistant'; content: string; sources?: number[] }

type Conversation = { id: string; title: string; time: string; messages: Message[] }

const sources: Source[] = [
  { id: 1, title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks', authors: 'Patrick Lewis, Ethan Perez, Aleksandandra Piktus et al.', year: '2020', tag: 'Foundational', excerpt: 'RAG combines a pretrained parametric memory with a non-parametric memory, an index of dense vectors built from a knowledge corpus.' },
  { id: 2, title: 'Precise Zero-Shot Dense Retrieval without Relevance Labels', authors: 'Akari Asai, Xinyan Yu, Jungo Kasai et al.', year: '2022', tag: 'Retrieval', excerpt: 'Contriever demonstrates that unsupervised contrastive learning can produce strong retrievers without annotated relevance labels.' },
  { id: 3, title: 'Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection', authors: 'Akari Asai, Zeqiu Wu, Yizhong Wang et al.', year: '2023', tag: 'Generation', excerpt: 'Self-RAG trains a single LM to adaptively retrieve passages and generate text, while reflecting on its own generation.' },
  { id: 4, title: 'From Local to Global: A Graph RAG Approach to Query-Focused Summarization', authors: 'Darren Edge, Ha Trinh, Newman Cheng et al.', year: '2024', tag: 'GraphRAG', excerpt: 'Graph-based indexing supports global sensemaking over private corpora by identifying themes and communities across the data.' },
]

const initialConversations: Conversation[] = [
  {
    id: 'graph', title: 'GraphRAG vs traditional RAG', time: 'Today', messages: [
      { role: 'user', content: 'How does GraphRAG compare to traditional RAG?' },
      { role: 'assistant', content: 'GraphRAG and traditional RAG solve different parts of the retrieval problem. Traditional RAG retrieves semantically similar chunks directly from a vector index, while GraphRAG first builds a structured graph of entities, relationships, and community summaries.\n\nFor questions that require **local detail**, traditional RAG is often simpler and more efficient. For questions requiring **global synthesis** across a corpus, GraphRAG can connect evidence that may never appear together in a single chunk. [1] [4]\n\nThe practical trade-off is an indexing step and higher query complexity in exchange for better multi-hop reasoning and corpus-level summaries.', sources: [1, 4] },
    ]
  },
  { id: 'eval', title: 'RAG evaluation methods', time: 'Yesterday', messages: [{ role: 'user', content: 'What are the main RAG evaluation methods?' }, { role: 'assistant', content: 'RAG evaluation typically separates retrieval quality from generation quality. Key metrics include context recall, context precision, answer faithfulness, and answer relevance.\n\nA strong evaluation set combines automated judges with human review, especially for open-ended research questions.', sources: [1, 3] }] },
  { id: 'hybrid', title: 'Hybrid retrieval approaches', time: 'Aug 28', messages: [{ role: 'user', content: 'What are the latest approaches to hybrid retrieval?' }, { role: 'assistant', content: 'Hybrid retrieval combines dense and sparse signals to improve recall and robustness. Dense retrievers capture semantic similarity, while BM25-style methods preserve exact terminology and rare entities. [1] [2]', sources: [1, 2] }] },
  { id: 'agents', title: 'Agents and long context', time: 'Aug 24', messages: [{ role: 'user', content: 'How are agents changing RAG systems?' }, { role: 'assistant', content: 'Recent systems treat retrieval as an action an agent can plan, critique, and repeat. This makes the retrieval loop adaptive rather than a fixed pre-processing step.', sources: [3] }] },
]

function Logo({ compact = false }: { compact?: boolean }) {
  return <div className="flex items-center gap-2.5"><div className="logo-mark"><span /><span /><span /></div>{!compact && <span className="text-[15px] font-semibold tracking-[-0.02em]">RAGWise</span>}</div>
}

function Sidebar({ conversations, activeId, onSelect, onNew, onClose, isDark, onToggleTheme, onToggleCollapse, sidebarCollapsed }: { conversations: Conversation[]; activeId: string | null; onSelect: (id: string) => void; onNew: () => void; onClose?: () => void; isDark: boolean; onToggleTheme: () => void; onToggleCollapse?: () => void; sidebarCollapsed?: boolean }) {
  return <aside className={`sidebar-panel flex h-full w-[274px] shrink-0 flex-col border-r border-border bg-sidebar px-3 py-4 ${sidebarCollapsed ? 'is-collapsed' : ''}`}>
    <div className="mb-8 flex items-center justify-between px-2"><Logo /><div className="flex items-center gap-1"><button onClick={onToggleCollapse} className="icon-button desktop-only-control" aria-label={sidebarCollapsed ? 'Expand navigation' : 'Collapse navigation'}><Menu size={18} /></button><button onClick={onClose} className="icon-button mobile-only-control" aria-label="Close navigation"><X size={18} /></button></div></div>
    <button className="new-chat mb-7 flex items-center gap-2.5 rounded-lg px-3.5 py-2.5 text-sm font-medium" onClick={onNew}><span className="new-chat-icon" aria-hidden="true">+</span><span>New chat</span></button>
    <div className="mb-2 flex items-center justify-between px-2 text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground"><span>Recent chats</span><button className="text-muted-foreground hover:text-foreground" aria-label="Search chats"><Search size={14} /></button></div>
    <nav className="flex flex-1 flex-col gap-0.5 overflow-auto">{conversations.map((conversation) => <button key={conversation.id} onClick={() => { onSelect(conversation.id); onClose?.() }} className={`history-item flex items-center gap-2 rounded-lg px-2.5 py-2.5 text-left text-[13px] ${activeId === conversation.id ? 'active' : ''}`}><MessageSquare size={15} className="shrink-0 text-muted-foreground" /><span className="truncate">{conversation.title}</span></button>)}</nav>
    <div className="sidebar-footer mt-5 border-t border-border pt-3"><button className="sidebar-link"><Settings2 size={16} />Settings</button><button className="sidebar-link" onClick={onToggleTheme} aria-label={`Switch to ${isDark ? 'light' : 'dark'} theme`}><div className="theme-icon">{isDark ? <Moon size={14} /> : <Sun size={14} />}</div>Theme <span className="theme-name">{isDark ? 'Dark' : 'Light'}</span><ChevronDown size={14} className="ml-auto text-muted-foreground" /></button><div className="mt-4 flex items-center gap-2.5 rounded-lg px-2.5 py-2"><div className="avatar">AR</div><div className="min-w-0"><p className="truncate text-xs font-medium">Alex Rivera</p><p className="text-[11px] text-muted-foreground">Pro workspace</p></div><MoreHorizontal size={16} className="ml-auto text-muted-foreground" /></div></div>
  </aside>
}

function SourcesPanel({ activeSources, onClose }: { activeSources: Source[]; onClose: () => void }) {
  return <aside className="sources-panel flex h-full w-[312px] shrink-0 flex-col border-l border-border bg-card"><div className="flex items-center justify-between border-b border-border px-5 py-4"><div><div className="flex items-center gap-2"><BookOpen size={16} className="text-primary" /><h2 className="text-sm font-semibold">Sources</h2></div><p className="mt-1 text-xs text-muted-foreground">{activeSources.length} papers used in this answer</p></div><button onClick={onClose} className="icon-button" aria-label="Close sources"><X size={17} /></button></div><div className="flex-1 overflow-auto px-4 py-4">{activeSources.map((source, index) => <article key={source.id} className="source-card group mb-3 rounded-xl border border-border bg-background p-3.5"><div className="mb-3 flex items-start justify-between"><span className="citation-badge">[{index + 1}]</span><span className="rounded-full bg-muted px-2 py-1 text-[10px] font-medium text-muted-foreground">{source.tag}</span></div><h3 className="text-[13px] font-semibold leading-5 text-foreground">{source.title}</h3><p className="mt-2 text-[11px] leading-4 text-muted-foreground">{source.authors}</p><div className="mt-3 flex items-center justify-between border-t border-border pt-3 text-[11px] text-muted-foreground"><span>{source.year}</span><button className="font-medium text-primary opacity-0 transition-opacity group-hover:opacity-100">View paper ↗</button></div><p className="mt-3 border-l-2 border-primary/30 pl-3 text-xs leading-5 text-muted-foreground">{source.excerpt}</p></article>)}</div></aside>
}

function MarkdownText({ content }: { content: string }) { return <div className="message-copy">{content.split('\n\n').map((paragraph, index) => <p key={index}>{paragraph.split(/(\[\d+\]|\*\*[^*]+\*\*)/g).map((part, i) => part.match(/^\[\d+\]$/) ? <button key={i} className="citation-link">{part}</button> : part.startsWith('**') ? <strong key={i}>{part.slice(2, -2)}</strong> : part)}</p>)}</div> }

export default function Page() {
  const [conversations, setConversations] = useState(initialConversations)
  const [activeId, setActiveId] = useState<string | null>('graph')
  const [input, setInput] = useState('')
  const [showSources, setShowSources] = useState(true)
  const [showSidebar, setShowSidebar] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [isDark, setIsDark] = useState(true)
  const active = conversations.find((conversation) => conversation.id === activeId)
  const activeSources = useMemo(() => (active?.messages.flatMap((message) => message.sources ?? []) ?? []).map((id) => sources.find((source) => source.id === id)).filter(Boolean) as Source[], [active])
  const send = (text = input) => { const trimmed = text.trim(); if (!trimmed) return; const id = `chat-${Date.now()}`; const newConversation: Conversation = { id, title: trimmed.length > 34 ? `${trimmed.slice(0, 34)}��` : trimmed, time: 'Just now', messages: [{ role: 'user', content: trimmed }, { role: 'assistant', content: 'I’m searching the RAGWise research collection for relevant evidence.\n\nThis demo response would synthesize findings from the most relevant papers, compare methods, and link each claim back to its source. [1] [2]', sources: [1, 2] }] }; setConversations((current) => [newConversation, ...current]); setActiveId(id); setInput(''); setShowSources(true) }
  const newChat = () => { setActiveId(null); setInput(''); setShowSources(false); setShowSidebar(false) }
  return <div className={isDark ? 'dark app-shell' : 'app-shell'}><div className="app-frame"><div className={`mobile-overlay ${showSidebar || showSources ? 'visible' : ''}`} onClick={() => { setShowSidebar(false); setShowSources(false) }} /><div className={`sidebar-drawer ${showSidebar ? 'open' : ''}`}><Sidebar conversations={conversations} activeId={activeId} onSelect={setActiveId} onNew={newChat} onClose={() => setShowSidebar(false)} isDark={isDark} onToggleTheme={() => setIsDark((dark) => !dark)} onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)} sidebarCollapsed={sidebarCollapsed} /></div><div className={`desktop-sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}><Sidebar conversations={conversations} activeId={activeId} onSelect={setActiveId} onNew={newChat} isDark={isDark} onToggleTheme={() => setIsDark((dark) => !dark)} onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)} sidebarCollapsed={sidebarCollapsed} /></div><main className="chat-area flex min-w-0 flex-1 flex-col"><header className="chat-header flex items-center justify-between border-b border-border px-4 py-3 md:px-7"><button className="icon-button mobile-only-control" onClick={() => setShowSidebar(true)} aria-label="Open navigation"><Menu size={19} /></button><div className="ml-auto flex items-center gap-1.5"><button className={`sources-toggle flex items-center gap-2 rounded-lg px-2.5 py-2 text-xs font-medium ${showSources ? 'selected' : ''}`} onClick={() => setShowSources(!showSources)}><PanelRight size={16} /> <span className="hidden sm:inline">Sources</span></button></div></header><div className="chat-scroll flex-1 overflow-auto"><div className={`chat-content mx-auto w-full ${active ? 'has-messages' : ''}`}>{!active ? <Welcome onPrompt={send} /> : <><div className="message-list">{active.messages.map((message, index) => <div key={index} className={`message-row ${message.role}`}>{message.role === 'assistant' && <div className="assistant-mark"><Sparkles size={15} /></div>}<div className={message.role === 'user' ? 'user-bubble' : 'assistant-bubble'}>{message.role === 'assistant' ? <MarkdownText content={message.content} /> : <p>{message.content}</p>}{message.role === 'assistant' && <div className="response-footer"><span><Check size={13} /> Answer grounded in {message.sources?.length} papers</span><button><MoreHorizontal size={15} /></button></div>}</div></div>)}</div></>}</div></div><div className="composer-wrap mx-auto w-full"><div className="composer"><textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing && event.keyCode !== 229) { event.preventDefault(); send() } }} placeholder="Ask a question about RAG research..." rows={1} aria-label="Ask a research question" /><div className="composer-bottom"><div className="flex items-center gap-1.5 text-[11px] text-muted-foreground"><button className="composer-tool"><Globe2 size={14} /> Research mode</button><span className="hidden sm:inline text-muted-foreground/40">•</span><span className="hidden sm:inline">3,000 papers indexed</span></div><button onClick={() => send()} className="send-button" disabled={!input.trim()} aria-label="Send message"><ArrowUp size={17} /></button></div></div><p className="disclaimer">RAGWise can make mistakes. Verify important findings with the original papers.</p></div></main><div className={`sources-drawer ${showSources ? 'open' : ''}`}><SourcesPanel activeSources={activeSources.length ? activeSources : sources.slice(0, 2)} onClose={() => setShowSources(false)} /></div></div></div>
}

function Welcome({ onPrompt }: { onPrompt: (text: string) => void }) { const prompts = ['What are the latest approaches to RAG?', 'What are the main RAG evaluation methods?']; return <div className="welcome"><div className="welcome-icon"><Sparkles size={20} /></div><p className="eyebrow">Your research copilot</p><h1>Explore the RAG<br /><span>research landscape.</span></h1><p className="welcome-subtitle">Ask questions, compare methods, and trace every answer back to the papers that support it.</p><div className="prompt-grid">{prompts.map((prompt) => <button key={prompt} onClick={() => onPrompt(prompt)} className="prompt-card"><span>{prompt}</span><ArrowUp size={15} /></button>)}</div></div> }
