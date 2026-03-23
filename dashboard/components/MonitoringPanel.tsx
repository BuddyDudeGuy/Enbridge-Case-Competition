'use client'

import { useState } from 'react'
import styles from './MonitoringPanel.module.css'
import { farms, fleetKpis, alerts, tdiHistory, subsystems, farmDetails } from '@/data/mockData'
import PredictiveChart from './PredictiveChart'
import EventTimeline from './EventTimeline'
import { eventChartData } from '@/data/eventChartData'

const severityLabel: Record<string, string> = {
  red: 'CRITICAL',
  amber: 'WARNING',
  green: 'RESOLVED',
}

function getTdiColor(tdi: number) {
  if (tdi >= 60) return 'red'
  if (tdi >= 30) return 'amber'
  return 'green'
}

interface MonitoringPanelProps {
  selectedFarm: string | null
  onBack: () => void
  onFarmSelect: (farmId: string) => void
}

export default function MonitoringPanel({ selectedFarm, onBack, onFarmSelect }: MonitoringPanelProps) {
  const [alertsExpanded, setAlertsExpanded] = useState(false)
  const maxTdi = Math.max(...tdiHistory.map((p) => p.tdi))

  if (selectedFarm) {
    const farm = farms.find((f) => f.id === selectedFarm)
    const details = farmDetails[selectedFarm]
    if (!farm || !details) return null

    const tdiColor = getTdiColor(farm.avgTdi)
    const strongSubs = subsystems.filter((s) => details.strongSubsystems.includes(s.name))

    return (
      <div className={styles.container}>
        {/* Farm Header */}
        <div className={styles.section}>
          <button className={styles.backButton} onClick={onBack}>
            &larr; Back to Fleet Overview
          </button>
          <div className={styles.farmHeader}>
            <div>
              <span className={styles.sectionTitle}>{farm.name} &mdash; {farm.location}</span>
              <div className={styles.farmHeaderMeta}>
                <span className={styles.statusLocation}>{farm.turbines} turbines</span>
                <span className={styles.statusTemp}>&Delta; {farm.tempDeviation}&deg;C</span>
              </div>
            </div>
            <span className={`${styles.statusTdi} ${styles[`tdi${tdiColor.charAt(0).toUpperCase() + tdiColor.slice(1)}`]}`}>
              TDI {farm.avgTdi.toFixed(1)}
            </span>
          </div>
        </div>

        {/* Detection Highlight */}
        <div className={styles.detectionBanner}>
          <span className={styles.sectionTitle}>DETECTION PERFORMANCE</span>
          <span className={styles.detectionText}>{details.detectionHighlight}</span>
        </div>

        {/* Key Event */}
        <div className={styles.keyEventCard}>
          <span className={styles.keyEventLabel}>KEY EVENT</span>
          <span className={styles.keyEventText}>{details.keyEvent}</span>
        </div>

        {/* Predictive Chart & Timeline */}
        {eventChartData[selectedFarm] && (
          <>
            <PredictiveChart
              title={eventChartData[selectedFarm].title}
              subtitle={eventChartData[selectedFarm].subtitle}
              subsystem={eventChartData[selectedFarm].subsystem}
              r2={eventChartData[selectedFarm].r2}
              warningDays={eventChartData[selectedFarm].warningDays}
              totalDays={eventChartData[selectedFarm].totalDays}
              data={eventChartData[selectedFarm].data}
              faultStart={eventChartData[selectedFarm].faultStart}
            />
            <EventTimeline
              warningDays={eventChartData[selectedFarm].warningDays}
              totalDays={eventChartData[selectedFarm].totalDays}
              eventLabel={farmDetails[selectedFarm].keyEvent}
            />
          </>
        )}

        {/* Prediction Accuracy */}
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}>PREDICTION ACCURACY</span>
          </div>
          <span className={styles.sectionSubtitle}>How well our model predicts each subsystem (R&sup2; mapped to 0-100)</span>
          <div className={styles.subsystemGrid}>
            {strongSubs.map((sub) => {
              const r2Key = sub.name === 'Generator Bearings' ? 'genBearingR2'
                : `${sub.name.toLowerCase()}R2`
              const r2 = details.keyMetrics[r2Key] ?? 0
              const r2Pct = Math.round(r2 * 100)
              const color = r2Pct >= 80 ? 'green' : r2Pct >= 60 ? 'amber' : 'red'
              return (
                <div key={sub.name} className={styles.subsystemCard}>
                  <div className={styles.subsystemGauge}>
                    <svg viewBox="0 0 36 36" className={styles.gaugeSvg}>
                      <path
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        fill="none"
                        stroke="rgba(0,0,0,0.06)"
                        strokeWidth="3"
                      />
                      <path
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        fill="none"
                        stroke={`var(--status-${color})`}
                        strokeWidth="3"
                        strokeDasharray={`${r2Pct}, 100`}
                        strokeLinecap="round"
                        className={styles.gaugeArc}
                      />
                    </svg>
                    <span className={styles.gaugeValue}>{r2Pct}</span>
                  </div>
                  <span className={styles.subsystemName}>{sub.name}</span>
                  <span className={styles.subsystemDev}>R&sup2; {r2.toFixed(2)}</span>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.container}>
      {/* System Status — now shows TDI per farm */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionTitle}>THERMAL DEGRADATION INDEX</span>
          <span className={styles.sectionTime}>LIVE</span>
        </div>
        <div className={styles.statusList}>
          {farms.map((farm) => (
            <div key={farm.id} className={styles.statusRow}>
              <span className={`status-dot ${farm.health}`} />
              <span className={styles.statusName}>{farm.name}</span>
              <span className={styles.statusLocation}>{farm.turbines} turbines</span>
              <span className={styles.statusTemp}>Δ {farm.tempDeviation}°C</span>
              <span className={`${styles.statusTdi} ${styles[`tdi${getTdiColor(farm.avgTdi).charAt(0).toUpperCase() + getTdiColor(farm.avgTdi).slice(1)}`]}`}>
                TDI {farm.avgTdi.toFixed(1)}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Metric Cards — TDI focused */}
      <div className={styles.metricsGrid}>
        <div className={styles.metricCard}>
          <span className={styles.metricLabel}>FLEET TDI</span>
          <div className={styles.metricRow}>
            <span className={`${styles.metricValue} ${styles.metricAmber}`}>{fleetKpis.fleetTdi}</span>
          </div>
          <span className={styles.metricSub}>0-100 composite score</span>
        </div>
        <div className={styles.metricCard}>
          <span className={styles.metricLabel}>CARE SCORE</span>
          <div className={styles.metricRow}>
            <span className={`${styles.metricValue} ${styles.metricGreen}`}>{fleetKpis.careScore}</span>
          </div>
          <span className={styles.metricSub}>detection accuracy</span>
        </div>
        <div className={styles.metricCard}>
          <span className={styles.metricLabel}>AVG TEMP DEVIATION</span>
          <div className={styles.metricRow}>
            <span className={styles.metricValue}>{fleetKpis.avgTempDeviation}</span>
            <span className={styles.metricUnit}>°C</span>
          </div>
          <span className={styles.metricSub}>above NBM prediction</span>
        </div>
        <div
          className={`${styles.metricCard} ${styles.metricCardClickable}`}
          onClick={() => setAlertsExpanded(!alertsExpanded)}
        >
          <span className={styles.metricLabel}>ACTIVE ALERTS {alertsExpanded ? '\u25B2' : '\u25BC'}</span>
          <div className={styles.metricRow}>
            <span className={`${styles.metricValue} ${styles.metricRed}`}>{fleetKpis.activeAlerts}</span>
          </div>
          <span className={styles.metricSub}>{fleetKpis.turbinesRed} critical · {fleetKpis.turbinesYellow} warning</span>
        </div>
      </div>

      {alertsExpanded && (
        <div className={styles.alertDropdown}>
          {alerts.filter(a => a.severity !== 'green').map((alert) => (
            <div
              key={alert.id}
              className={styles.alertDropdownItem}
              onClick={() => onFarmSelect(alert.farm)}
            >
              <span className={`${styles.feedSeverity} ${styles[`severity${alert.severity.charAt(0).toUpperCase() + alert.severity.slice(1)}`]}`}>
                {severityLabel[alert.severity]}
              </span>
              <span className={styles.alertDropdownFarm}>Farm {alert.farm}</span>
              <span className={styles.alertDropdownTurbine}>{alert.turbine}</span>
              <span className={styles.alertDropdownMessage}>{alert.message}</span>
              <span className={styles.alertDropdownArrow}>&rarr;</span>
            </div>
          ))}
        </div>
      )}

      {/* TDI Trend Chart */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionTitle}>FLEET TDI TREND — 12H</span>
          <span className={styles.sectionSub}>score 0-100</span>
        </div>
        <div className={styles.chartArea}>
          <div className={styles.chartThresholds}>
            <div className={styles.thresholdLine} style={{ bottom: '60%' }}>
              <span className={styles.thresholdLabel}>60 — Critical</span>
            </div>
            <div className={styles.thresholdLine} style={{ bottom: '30%' }}>
              <span className={styles.thresholdLabel}>30 — Warning</span>
            </div>
          </div>
          <div className={styles.chart}>
            {tdiHistory.map((p, i) => (
              <div key={i} className={styles.barCol}>
                <div className={styles.barWrapper}>
                  <div
                    className={`${styles.bar} ${styles[`bar${getTdiColor(p.tdi).charAt(0).toUpperCase() + getTdiColor(p.tdi).slice(1)}`]}`}
                    style={{
                      height: `${(p.tdi / 100) * 100}%`,
                      animationDelay: `${i * 0.05}s`,
                    }}
                  />
                </div>
                <span className={styles.barLabel}>
                  {i % 2 === 0 ? p.hour : ''}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Activity Feed */}
      <div className={`${styles.section} ${styles.sectionDark}`}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionTitle}>ANOMALY FEED</span>
          <div className={styles.liveBadge}>
            <span className={styles.liveDot} />
            <span>LIVE</span>
          </div>
        </div>
        <div className={styles.feed}>
          {alerts.map((alert) => (
            <div key={alert.id} className={styles.feedItem}>
              <div className={`${styles.feedSeverity} ${styles[`severity${alert.severity.charAt(0).toUpperCase() + alert.severity.slice(1)}`]}`}>
                {severityLabel[alert.severity]}
              </div>
              <div className={styles.feedContent}>
                <div className={styles.feedMeta}>
                  <span className={styles.feedTime}>{alert.time}</span>
                  <span className={styles.feedFarm}>Farm {alert.farm}</span>
                  <span className={styles.feedTurbine}>{alert.turbine}</span>
                  <span className={styles.feedSubsystem}>{alert.subsystem}</span>
                </div>
                <span className={styles.feedMessage}>{alert.message}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Subsystem Health */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionTitle}>PREDICTION ACCURACY</span>
        </div>
        <span className={styles.sectionSubtitle}>Model prediction confidence per subsystem (higher = better)</span>
        <div className={styles.subsystemGrid}>
          {subsystems.map((sub) => {
            const color = sub.health >= 90 ? 'green' : sub.health >= 75 ? 'amber' : 'red'
            const hasLimitation = 'limitation' in sub && !!(sub as any).limitation
            return (
              <div key={sub.name} className={`${styles.subsystemCard} ${hasLimitation ? styles.subsystemLimited : ''}`}>
                {hasLimitation ? (
                  <div className={styles.limitationBadge}>
                    <span className={styles.limitationIcon}>&#9889;</span>
                  </div>
                ) : (
                  <div className={styles.subsystemGauge}>
                    <svg viewBox="0 0 36 36" className={styles.gaugeSvg}>
                      <path
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        fill="none"
                        stroke="rgba(0,0,0,0.06)"
                        strokeWidth="3"
                      />
                      <path
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        fill="none"
                        stroke={`var(--status-${color})`}
                        strokeWidth="3"
                        strokeDasharray={`${sub.health}, 100`}
                        strokeLinecap="round"
                        className={styles.gaugeArc}
                      />
                    </svg>
                    <span className={styles.gaugeValue}>{sub.health}</span>
                  </div>
                )}
                <span className={styles.subsystemName}>{sub.name}</span>
                {hasLimitation ? (
                  <span className={styles.limitationText}>{(sub as any).limitation}</span>
                ) : (
                  <span className={styles.subsystemDev}>Score {sub.health}</span>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
