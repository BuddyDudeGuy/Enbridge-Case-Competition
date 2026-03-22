import styles from './page.module.css'
import Header from '@/components/Header'
import Globe from '@/components/Globe'
import MonitoringPanel from '@/components/MonitoringPanel'
import ChatDock from '@/components/ChatDock'

export default function Home() {
  return (
    <div className={styles.container}>
      <Header />
      <main className={styles.main}>
        <div className={styles.globePanel}>
          <Globe />
        </div>
        <div className={styles.monitorPanel}>
          <MonitoringPanel />
        </div>
      </main>
      <footer className={styles.chatDock}>
        <ChatDock />
      </footer>
    </div>
  )
}
