'use client'

import { useEffect, useRef, useState } from 'react'
import createGlobe from 'cobe'
import { useSpring } from 'react-spring'
import styles from './Globe.module.css'
import { farms } from '@/data/mockData'

const healthColor: Record<string, [number, number, number]> = {
  green: [0.13, 0.77, 0.37],
  amber: [0.96, 0.62, 0.04],
  red: [0.94, 0.27, 0.27],
}

export default function Globe() {
  const canvasRef = useRef<HTMLCanvasElement>(null!)
  const pointerInteracting = useRef<number | null>(null)
  const pointerInteractionMovement = useRef(0)
  const phiRef = useRef(0)
  const [hoveredFarm, setHoveredFarm] = useState<string | null>(null)

  const [{ r }, api] = useSpring(() => ({
    r: 0,
    config: { mass: 1, tension: 280, friction: 40, precision: 0.001 },
  }))

  useEffect(() => {
    let width = 0
    const onResize = () => {
      if (canvasRef.current) {
        width = canvasRef.current.offsetWidth
      }
    }
    onResize()
    window.addEventListener('resize', onResize)

    const globe = createGlobe(canvasRef.current, {
      devicePixelRatio: 2,
      width: width * 2,
      height: width * 2,
      phi: 0,
      theta: 0.15,
      dark: 0,
      diffuse: 1.4,
      mapSamples: 16000,
      mapBrightness: 5.5,
      baseColor: [0.92, 0.92, 0.94],
      markerColor: [0.13, 0.77, 0.37],
      glowColor: [0.95, 0.95, 0.97],
      markers: farms.flatMap((farm) => [
        // Outer glow ring
        {
          location: [farm.lat, farm.lng] as [number, number],
          size: 0.18,
          color: healthColor[farm.health].map((c) => c * 0.3) as [number, number, number],
        },
        // Mid glow ring
        {
          location: [farm.lat, farm.lng] as [number, number],
          size: 0.12,
          color: healthColor[farm.health].map((c) => c * 0.6) as [number, number, number],
        },
        // Core marker
        {
          location: [farm.lat, farm.lng] as [number, number],
          size: 0.07,
          color: healthColor[farm.health],
        },
      ]),
      onRender: (state) => {
        if (!pointerInteracting.current) {
          phiRef.current += 0.003
        }
        state.phi = phiRef.current + r.get()
        state.width = width * 2
        state.height = width * 2
      },
    })

    return () => {
      globe.destroy()
      window.removeEventListener('resize', onResize)
    }
  }, [r])

  return (
    <div className={styles.container}>
      <div className={styles.globeTitle}>
        <span className={styles.titleLabel}>Real-time SCADA Feed</span>
      </div>

      <div className={styles.globeWrapper}>
        <canvas
          ref={canvasRef}
          className={styles.canvas}
          onPointerDown={(e) => {
            pointerInteracting.current = e.clientX - pointerInteractionMovement.current
            if (canvasRef.current) canvasRef.current.style.cursor = 'grabbing'
          }}
          onPointerUp={() => {
            pointerInteracting.current = null
            if (canvasRef.current) canvasRef.current.style.cursor = 'grab'
          }}
          onPointerOut={() => {
            pointerInteracting.current = null
            if (canvasRef.current) canvasRef.current.style.cursor = 'grab'
          }}
          onMouseMove={(e) => {
            if (pointerInteracting.current !== null) {
              const delta = e.clientX - pointerInteracting.current
              pointerInteractionMovement.current = delta
              api.start({ r: delta / 200 })
            }
          }}
          onTouchMove={(e) => {
            if (pointerInteracting.current !== null && e.touches[0]) {
              const delta = e.touches[0].clientX - pointerInteracting.current
              pointerInteractionMovement.current = delta
              api.start({ r: delta / 100 })
            }
          }}
        />
      </div>

      <div className={styles.stats}>
        <div className={styles.statItem}>
          <span className={styles.statDot}>●</span>
          <span className={styles.statValue}>3</span>
          <span className={styles.statLabel}>Wind Farms</span>
        </div>
        <div className={styles.statSep}>|</div>
        <div className={styles.statItem}>
          <span className={styles.statValue}>36</span>
          <span className={styles.statLabel}>Turbines</span>
        </div>
        <div className={styles.statSep}>|</div>
        <div className={styles.statItem}>
          <span className={styles.statValue}>135.3</span>
          <span className={styles.statLabel}>MW Output</span>
        </div>
        <div className={styles.statSep}>|</div>
        <div className={styles.statItem}>
          <span className={styles.statValue}>87%</span>
          <span className={styles.statLabel}>Fleet Health</span>
        </div>
      </div>

      <div className={styles.legend}>
        <div className={styles.legendRow}>
          <span className={`status-dot green`} />
          <span className={styles.legendLabel}>Online</span>
          <span className={`status-dot amber`} />
          <span className={styles.legendLabel}>Warning</span>
          <span className={`status-dot red`} />
          <span className={styles.legendLabel}>Critical</span>
        </div>
      </div>

      <div className={styles.farmCards}>
        {farms.map((farm) => (
          <div
            key={farm.id}
            className={`${styles.farmCard} ${hoveredFarm === farm.id ? styles.farmCardHover : ''}`}
            onMouseEnter={() => setHoveredFarm(farm.id)}
            onMouseLeave={() => setHoveredFarm(null)}
          >
            <div className={styles.farmCardHeader}>
              <span className={`status-dot ${farm.health}`} />
              <span className={styles.farmName}>{farm.name}</span>
            </div>
            <div className={styles.farmCardBody}>
              <div className={styles.farmStat}>
                <span className={styles.farmStatValue}>{farm.turbines}</span>
                <span className={styles.farmStatLabel}>turbines</span>
              </div>
              <div className={styles.farmStat}>
                <span className={styles.farmStatValue}>{farm.windSpeed}</span>
                <span className={styles.farmStatLabel}>m/s</span>
              </div>
              <div className={styles.farmStat}>
                <span className={styles.farmStatValue}>{farm.avgTdi.toFixed(1)}</span>
                <span className={styles.farmStatLabel}>TDI</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
