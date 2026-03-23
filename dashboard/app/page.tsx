'use client'

import { useState } from 'react'
import styles from './page.module.css'
import Header from '@/components/Header'
import Globe from '@/components/Globe'
import MonitoringPanel from '@/components/MonitoringPanel'

export default function Home() {
  const [selectedFarm, setSelectedFarm] = useState<string | null>(null)

  return (
    <div className={styles.container}>
      <Header />
      <main className={styles.main}>
        <div className={styles.globePanel}>
          <Globe onFarmSelect={setSelectedFarm} selectedFarm={selectedFarm} />
        </div>
        <div className={styles.monitorPanel}>
          <MonitoringPanel selectedFarm={selectedFarm} onBack={() => setSelectedFarm(null)} onFarmSelect={setSelectedFarm} />
        </div>
      </main>
    </div>
  )
}
