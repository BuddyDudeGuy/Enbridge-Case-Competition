'use client'

import styles from './MonitoringPanel.module.css'
import { farms, fleetKpis, alerts, tdiHistory, subsystems } from '@/data/mockData'

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

export default function MonitoringPanel() {
  const maxTdi = Math.max(...tdiHistory.map((p) => p.tdi))

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
        <div className={styles.metricCard}>
          <span className={styles.metricLabel}>ACTIVE ALERTS</span>
          <div className={styles.metricRow}>
            <span className={`${styles.metricValue} ${styles.metricRed}`}>{fleetKpis.activeAlerts}</span>
          </div>
          <span className={styles.metricSub}>{fleetKpis.turbinesRed} critical · {fleetKpis.turbinesYellow} warning</span>
        </div>
      </div>

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
          <span className={styles.sectionTitle}>THERMAL SUBSYSTEM HEALTH</span>
        </div>
        <div className={styles.subsystemGrid}>
          {subsystems.map((sub) => {
            const color = sub.health >= 90 ? 'green' : sub.health >= 75 ? 'amber' : 'red'
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
                      strokeDasharray={`${sub.health}, 100`}
                      strokeLinecap="round"
                      className={styles.gaugeArc}
                    />
                  </svg>
                  <span className={styles.gaugeValue}>{sub.health}</span>
                </div>
                <span className={styles.subsystemName}>{sub.name}</span>
                <span className={styles.subsystemDev}>Δ {sub.avgDeviation}°C</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
