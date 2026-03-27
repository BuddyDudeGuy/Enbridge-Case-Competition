'use client'

import { useState } from 'react'
import styles from './EventTimeline.module.css'

interface EventTimelineProps {
  warningDays: number
  totalDays: number
  eventLabel: string
}

export default function EventTimeline({
  warningDays,
  totalDays,
  eventLabel,
}: EventTimelineProps) {
  const [showInfo, setShowInfo] = useState(false)
  const faultDuration = Math.max(Math.round(totalDays * 0.1), 1)
  const normalDays = Math.max(0, totalDays - warningDays - faultDuration)
  const normalPct = (normalDays / totalDays) * 100
  const warningPct = (warningDays / totalDays) * 100
  const faultPct = (faultDuration / totalDays) * 100

  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <div className={styles.titleRow}>
          <span className={styles.title}>DETECTION TIMELINE</span>
          <button
            className={`${styles.infoBtn} ${showInfo ? styles.infoBtnActive : ''}`}
            onClick={() => setShowInfo((v) => !v)}
            aria-label="Toggle timeline info"
          >
            &#x2139;
          </button>
        </div>
        <span className={styles.eventLabel}>{eventLabel}</span>
      </div>

      {showInfo && (
        <div className={styles.infoBox}>
          This timeline shows when our model detected abnormal thermal behavior
          relative to the recorded fault. The amber &ldquo;early warning&rdquo; zone
          represents the lead time our system provides before the fault was officially
          recorded &mdash; time that could be used to schedule maintenance and avoid
          emergency repairs.
        </div>
      )}

      <div className={styles.timeline}>
        {/* Labels row */}
        <div className={styles.labels}>
          <span className={styles.label} style={{ width: `${normalPct}%` }}>
            Normal Operation
          </span>
          <span className={styles.label} style={{ width: `${warningPct}%` }}>
            {warningDays}d early warning
          </span>
          <span className={styles.label} style={{ width: `${faultPct}%` }}>
            Fault
          </span>
        </div>

        {/* Bar */}
        <div className={styles.bar}>
          <div className={styles.segmentNormal} style={{ width: `${normalPct}%` }}>
            <div className={styles.marker}>
              <span className={styles.markerArrow} />
            </div>
          </div>
          <div className={styles.segmentWarning} style={{ width: `${warningPct}%` }}>
            <div className={styles.marker}>
              <span className={styles.markerArrow} />
            </div>
          </div>
          <div className={styles.segmentFault} style={{ width: `${faultPct}%` }} />
        </div>

        {/* Day markers */}
        <div className={styles.dayMarkers}>
          <span className={styles.dayMark}>Day 0</span>
          {normalPct > 8 && (
            <span className={styles.dayMark} style={{ left: `${normalPct}%` }}>
              Day {normalDays}
            </span>
          )}
          {normalPct + warningPct > 8 && normalPct + warningPct < 92 && (
            <span className={styles.dayMark} style={{ left: `${normalPct + warningPct}%` }}>
              Day {normalDays + warningDays}
            </span>
          )}
          <span className={styles.dayMarkEnd}>Day {totalDays}</span>
        </div>
      </div>
    </div>
  )
}
