'use client'

import dynamic from 'next/dynamic'

const ChatPage = dynamic(() => import('./chat-page'), {
  ssr: false,
  loading: () => <div className="auth-loading"><span>Connecting to RAGWise…</span></div>,
})

export default function Page() {
  return <ChatPage />
}
