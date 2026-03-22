'use client'

import styles from './Header.module.css'
import { fleetKpis } from '@/data/mockData'

export default function Header() {
  return (
    <header className={styles.header}>
      <div className={styles.brand}>
        <div className={styles.logoIcon}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </svg>
        </div>
        <div className={styles.brandText}>
          <span className={styles.logo}>TURBINE MONITORING SYSTEMS</span>
          <span className={styles.subtitle}>ENBRIDGE WIND ENERGY</span>
        </div>
        <span className={styles.version}>v2.4</span>
      </div>

      <div className={styles.kpis}>
        <div className={styles.kpiChip}>
          <span className={styles.kpiIcon}>⊞</span>
          <span className={styles.kpiValue}>{fleetKpis.totalTurbines}</span>
          <span className={styles.kpiLabel}>Turbines</span>
        </div>

        <div className={styles.divider} />

        <div className={styles.kpiChip}>
          <span className={`${styles.kpiIcon} ${styles.alertIcon}`}>●</span>
          <span className={styles.kpiValue}>{fleetKpis.activeAlerts}</span>
          <span className={styles.kpiLabel}>Alerts</span>
        </div>

        <div className={styles.divider} />

        <div className={styles.kpiChip}>
          <span className={styles.kpiIcon}>◈</span>
          <span className={styles.kpiValue}>{fleetKpis.fleetHealth}%</span>
          <span className={styles.kpiLabel}>Health</span>
        </div>

        <div className={styles.divider} />

        <div className={styles.kpiChip}>
          <span className={styles.kpiIcon}>↑</span>
          <span className={styles.kpiValue}>{fleetKpis.uptime}%</span>
          <span className={styles.kpiLabel}>Uptime</span>
        </div>

        <div className={styles.divider} />

        <div className={`${styles.kpiChip} ${styles.careChip}`}>
          <span className={styles.kpiLabel}>CARE</span>
          <span className={styles.kpiValue}>{fleetKpis.careScore}</span>
        </div>
      </div>

      <div className={styles.meta}>
        <div className={styles.connectionStatus}>
          <span className={styles.connDot} />
          <span className={styles.connLabel}>SCADA CONNECTED</span>
        </div>
        <span className={styles.time}>
          Mar 22, 2026 · 14:32 UTC
        </span>
      </div>
    </header>
  )
}
