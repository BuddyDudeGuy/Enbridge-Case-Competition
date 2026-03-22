'use client'

import styles from './MonitoringPanel.module.css'
import { farms, fleetKpis, alerts, powerHistory, subsystems } from '@/data/mockData'

const severityLabel: Record<string, string> = {
  red: 'CRITICAL',
  amber: 'WARNING',
  green: 'RESOLVED',
}

export default function MonitoringPanel() {
  const maxMw = Math.max(...powerHistory.map((p) => p.mw))

  return (
    <div className={styles.container}>
      {/* System Status */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionTitle}>SYSTEM STATUS</span>
          <span className={styles.sectionTime}>LIVE</span>
        </div>
        <div className={styles.statusList}>
          {farms.map((farm) => (
            <div key={farm.id} className={styles.statusRow}>
              <span className={`status-dot ${farm.health}`} />
              <span className={styles.statusName}>{farm.name}</span>
              <span className={styles.statusLocation}>{farm.location}</span>
              <span className={styles.statusWind}>{farm.windSpeed} m/s</span>
              <span className={styles.statusPower}>{farm.powerOutput} MW</span>
            </div>
          ))}
        </div>
      </div>

      {/* Metric Cards */}
      <div className={styles.metricsGrid}>
        <div className={styles.metricCard}>
          <span className={styles.metricLabel}>TOTAL OUTPUT</span>
          <div className={styles.metricRow}>
            <span className={styles.metricValue}>{fleetKpis.totalOutput}</span>
            <span className={styles.metricUnit}>MW</span>
          </div>
        </div>
        <div className={styles.metricCard}>
          <span className={styles.metricLabel}>AVG WIND SPEED</span>
          <div className={styles.metricRow}>
            <span className={styles.metricValue}>{fleetKpis.avgWindSpeed}</span>
            <span className={styles.metricUnit}>m/s</span>
          </div>
        </div>
        <div className={styles.metricCard}>
          <span className={styles.metricLabel}>FLEET UPTIME</span>
          <div className={styles.metricRow}>
            <span className={`${styles.metricValue} ${styles.metricGreen}`}>{fleetKpis.uptime}%</span>
          </div>
        </div>
        <div className={styles.metricCard}>
          <span className={styles.metricLabel}>ACTIVE ALERTS</span>
          <div className={styles.metricRow}>
            <span className={`${styles.metricValue} ${styles.metricRed}`}>{fleetKpis.activeAlerts}</span>
          </div>
        </div>
      </div>

      {/* Power Output Chart */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionTitle}>POWER OUTPUT — 12H</span>
          <span className={styles.sectionSub}>MW</span>
        </div>
        <div className={styles.chart}>
          {powerHistory.map((p, i) => (
            <div key={i} className={styles.barCol}>
              <div className={styles.barWrapper}>
                <div
                  className={styles.bar}
                  style={{
                    height: `${(p.mw / maxMw) * 100}%`,
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

      {/* Activity Feed */}
      <div className={`${styles.section} ${styles.sectionDark}`}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionTitle}>ACTIVITY FEED</span>
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
          <span className={styles.sectionTitle}>SUBSYSTEM HEALTH</span>
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
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
