'use client'

import { useCallback, useLayoutEffect, useRef, useState } from 'react'

export function useChatScroll(conversationId: string | null) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const following = useRef(true)
  const prepend = useRef<{ id: string; height: number; top: number } | null>(null)
  const anchor = useRef<{ element: HTMLElement; offset: number } | null>(null)
  const [showJump, setShowJump] = useState(false)
  const rememberAnchor = useCallback(() => {
    const scroll = scrollRef.current
    if (!scroll) return
    const top = scroll.getBoundingClientRect().top
    const element = Array.from(scroll.querySelectorAll<HTMLElement>('[data-message-id]')).find((message) => message.getBoundingClientRect().bottom > top)
    anchor.current = element ? { element, offset: element.getBoundingClientRect().top - top } : null
  }, [])
  const jumpToLatest = useCallback(() => {
    following.current = true
    anchor.current = null
    const scroll = scrollRef.current
    if (scroll) scroll.scrollTop = scroll.scrollHeight
    setShowJump(false)
  }, [])
  const followLatest = useCallback(() => { following.current = true; anchor.current = null }, [])
  const onScroll = useCallback(() => {
    const scroll = scrollRef.current
    if (!scroll) return
    following.current = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight <= 80
    setShowJump(!following.current)
    rememberAnchor()
  }, [rememberAnchor])
  const preparePrepend = useCallback((id: string) => {
    const scroll = scrollRef.current
    if (scroll && id === conversationId) {
      prepend.current = { id, height: scroll.scrollHeight, top: scroll.scrollTop }
      following.current = false
      rememberAnchor()
    }
  }, [conversationId, rememberAnchor])
  const adjust = useCallback(() => {
    const scroll = scrollRef.current
    if (!scroll) return
    if (following.current) scroll.scrollTop = scroll.scrollHeight
    else {
      const saved = anchor.current
      if (saved?.element.isConnected) scroll.scrollTop += saved.element.getBoundingClientRect().top - scroll.getBoundingClientRect().top - saved.offset
      setShowJump(scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight > 80)
    }
  }, [])
  useLayoutEffect(() => {
    const scroll = scrollRef.current
    if (!scroll) return
    const previous = prepend.current
    if (previous) {
      if (previous.id === conversationId) {
        scroll.scrollTop = previous.top + scroll.scrollHeight - previous.height
        adjust()
      }
      prepend.current = null
      onScroll()
    } else adjust()
  })
  useLayoutEffect(() => {
    const scroll = scrollRef.current
    const content = contentRef.current
    if (!scroll || !content) return
    const observer = new ResizeObserver(adjust)
    observer.observe(scroll)
    observer.observe(content)
    return () => observer.disconnect()
  })
  return { scrollRef, contentRef, onScroll, jumpToLatest, showJump, preparePrepend, followLatest }
}
