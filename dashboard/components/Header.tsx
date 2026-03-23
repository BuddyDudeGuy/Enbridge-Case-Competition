'use client'

import styles from './Header.module.css'
import { fleetKpis } from '@/data/mockData'

export default function Header() {
  return (
    <header className={styles.header}>
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
          <span className={styles.kpiValue}>{fleetKpis.fleetTdi}</span>
          <span className={styles.kpiLabel}>Fleet TDI</span>
        </div>

        <div className={styles.divider} />

        <div className={styles.kpiChip}>
          <span className={styles.kpiIcon}>↑</span>
          <span className={styles.kpiValue}>{fleetKpis.avgTempDeviation}°C</span>
          <span className={styles.kpiLabel}>Avg Δ Temp</span>
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
