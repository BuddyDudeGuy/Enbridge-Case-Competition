'use client'

import { useState } from 'react'
import styles from './ChatDock.module.css'
import { chatConversation, suggestedQueries } from '@/data/mockData'

export default function ChatDock() {
  const [expanded, setExpanded] = useState(true)

  return (
    <div className={`${styles.dock} ${expanded ? styles.expanded : styles.collapsed}`}>
      <button className={styles.toggle} onClick={() => setExpanded(!expanded)}>
        <div className={styles.toggleLeft}>
          <div className={styles.aiIcon}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <span className={styles.toggleTitle}>TMS Fleet Assistant</span>
          <span className={styles.toggleBadge}>CONNECTED</span>
          <span className={styles.toggleContext}>3 FARMS IN CONTEXT</span>
        </div>
        <span className={styles.chevron}>{expanded ? '▾' : '▴'}</span>
      </button>

      {expanded && (
        <div className={styles.body}>
          <div className={styles.messages}>
            {chatConversation.map((msg, i) => (
              <div
                key={i}
                className={`${styles.message} ${msg.role === 'user' ? styles.userMsg : styles.aiMsg}`}
              >
                {msg.role === 'ai' && (
                  <div className={styles.msgAvatar}>AI</div>
                )}
                <div className={styles.msgBubble}>
                  <span className={styles.msgText}>
                    {msg.text.split('\n').map((line, j) => (
                      <span key={j}>
                        {line}
                        {j < msg.text.split('\n').length - 1 && <br />}
                      </span>
                    ))}
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div className={styles.inputArea}>
            <div className={styles.inputWrapper}>
              <input
                type="text"
                className={styles.input}
                placeholder="Ask about fleet health, anomalies, maintenance..."
                readOnly
              />
              <button className={styles.sendBtn}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </button>
            </div>
            <div className={styles.suggestions}>
              <span className={styles.suggestLabel}>Suggest:</span>
              {suggestedQueries.map((q, i) => (
                <button key={i} className={styles.suggestChip}>{q}</button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
