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

interface GlobeProps {
  onFarmSelect: (farmId: string) => void
  selectedFarm: string | null
}

export default function Globe({ onFarmSelect, selectedFarm }: GlobeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null!)
  const pointerInteracting = useRef<number | null>(null)
  const pointerInteractionMovement = useRef(0)
  const phiRef = useRef(0)
  const targetPhiRef = useRef<number | null>(null)
  const targetThetaRef = useRef<number | null>(null)
  const thetaRef = useRef(0.15)
  const selectedFarmRef = useRef<string | null>(null)
  const pointerDownPos = useRef<{ x: number; y: number } | null>(null)
  const [hoveredFarm, setHoveredFarm] = useState<string | null>(null)

  const [{ r }, api] = useSpring(() => ({
    r: 0,
    config: { mass: 1, tension: 280, friction: 40, precision: 0.001 },
  }))

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const canvasSize = rect.width
    const radius = canvasSize / 2
    const cx = e.clientX - rect.left - radius
    const cy = e.clientY - rect.top - radius

    // Check if click is on the sphere
    const dist2 = cx * cx + cy * cy
    if (dist2 > radius * radius) return

    const rNorm = Math.sqrt(dist2) / radius
    const cz = Math.sqrt(1 - rNorm * rNorm) * radius

    // Convert to unit sphere coords
    const ux = cx / radius
    const uy = -cy / radius // flip y: screen y is down, sphere y is up
    const uz = cz / radius

    // Spherical coordinates on the unit sphere
    const clickLat = Math.asin(uy) * (180 / Math.PI)
    // atan2(x, z) gives azimuth angle
    const clickLngRaw = Math.atan2(ux, uz) * (180 / Math.PI)

    // Account for current globe rotation
    // cobe state.phi = phiRef + r.get(), state.theta = thetaRef
    const currentPhi = phiRef.current + r.get()
    const currentTheta = thetaRef.current

    // phi is longitude rotation (radians), theta is latitude tilt (radians)
    const geoLng = clickLngRaw - currentPhi * (180 / Math.PI)
    const geoLat = clickLat + currentTheta * (180 / Math.PI)

    // Find nearest farm
    let bestFarm: string | null = null
    let bestDist = Infinity
    for (const farm of farms) {
      const dLat = farm.lat - geoLat
      const dLng = farm.lng - geoLng
      // Normalize longitude difference to [-180, 180]
      const dLngNorm = ((dLng + 540) % 360) - 180
      const angularDist = Math.sqrt(dLat * dLat + dLngNorm * dLngNorm)
      if (angularDist < bestDist) {
        bestDist = angularDist
        bestFarm = farm.id
      }
    }

    if (bestFarm && bestDist < 15) {
      onFarmSelect(bestFarm)
    }
  }

  // Handle globe rotation when selectedFarm changes
  useEffect(() => {
    selectedFarmRef.current = selectedFarm
    if (selectedFarm) {
      const farm = farms.find((f) => f.id === selectedFarm)
      if (farm) {
        // Convert lng to phi (cobe phi = longitude in radians, offset by pi)
        // We need to set phiRef so the globe rotates to show this longitude
        const targetPhi = (-farm.lng * Math.PI) / 180
        // Convert lat to theta (cobe theta: 0 = equator view, positive = looking from above)
        const targetTheta = (farm.lat * Math.PI) / 180

        targetPhiRef.current = targetPhi
        targetThetaRef.current = targetTheta

        // Reset spring interaction so it doesn't interfere
        pointerInteractionMovement.current = 0
        api.start({ r: 0 })
      }
    } else {
      targetPhiRef.current = null
      targetThetaRef.current = null
    }
  }, [selectedFarm, api])

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
      markers: farms.flatMap((farm) => {
        const isSelected = selectedFarmRef.current === farm.id
        const hasSelection = selectedFarmRef.current !== null
        const dimFactor = hasSelection && !isSelected ? 0.25 : 1
        const sizeMult = isSelected ? 1.6 : 1
        return [
          // Outer glow ring
          {
            location: [farm.lat, farm.lng] as [number, number],
            size: 0.24 * sizeMult,
            color: healthColor[farm.health].map((c) => c * 0.3 * dimFactor) as [number, number, number],
          },
          // Mid glow ring
          {
            location: [farm.lat, farm.lng] as [number, number],
            size: 0.16 * sizeMult,
            color: healthColor[farm.health].map((c) => c * 0.6 * dimFactor) as [number, number, number],
          },
          // Core marker
          {
            location: [farm.lat, farm.lng] as [number, number],
            size: 0.10 * sizeMult,
            color: healthColor[farm.health].map((c) => c * dimFactor) as [number, number, number],
          },
        ]
      }),
      onRender: (state) => {
        if (targetPhiRef.current !== null && targetThetaRef.current !== null) {
          // Smoothly animate toward the target rotation
          const dphi = targetPhiRef.current - phiRef.current
          const dtheta = targetThetaRef.current - thetaRef.current
          phiRef.current += dphi * 0.06
          thetaRef.current += dtheta * 0.06
        } else if (!pointerInteracting.current) {
          phiRef.current += 0.003
        }
        state.phi = phiRef.current + r.get()
        state.theta = thetaRef.current
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
      <div className={styles.farmCardsVertical}>
        {farms.map((farm) => {
          const isSelected = selectedFarm === farm.id
          const hasSel = selectedFarm !== null
          const cardClasses = [
            styles.farmCard,
            hoveredFarm === farm.id ? styles.farmCardHover : '',
            isSelected ? styles.farmCardSelected : '',
            hasSel && !isSelected ? styles.farmCardDimmed : '',
          ]
            .filter(Boolean)
            .join(' ')
          return (
          <div
            key={farm.id}
            className={cardClasses}
            onClick={() => onFarmSelect(farm.id)}
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
                <span className={styles.farmStatValue}>{farm.avgTdi.toFixed(1)}</span>
                <span className={styles.farmStatLabel}>TDI</span>
              </div>
            </div>
          </div>
          )
        })}
      </div>
      <div className={styles.globeArea}>
        <canvas
          ref={canvasRef}
          className={styles.canvas}
          onPointerDown={(e) => {
            pointerInteracting.current = e.clientX - pointerInteractionMovement.current
            pointerDownPos.current = { x: e.clientX, y: e.clientY }
            if (canvasRef.current) canvasRef.current.style.cursor = 'grabbing'
          }}
          onPointerUp={(e) => {
            pointerInteracting.current = null
            if (canvasRef.current) canvasRef.current.style.cursor = 'grab'
            // Detect click (not drag): movement < 5px
            if (pointerDownPos.current) {
              const dx = e.clientX - pointerDownPos.current.x
              const dy = e.clientY - pointerDownPos.current.y
              if (Math.sqrt(dx * dx + dy * dy) < 5) {
                handleCanvasClick(e)
              }
              pointerDownPos.current = null
            }
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

        <div className={styles.titleOverlay}>
          <span className={styles.titleLabel}>Real-time SCADA Feed</span>
        </div>

        <div className={styles.legendOverlay}>
          <div className={styles.legendRow}>
            <span className={`status-dot green`} />
            <span className={styles.legendLabel}>Online</span>
            <span className={`status-dot amber`} />
            <span className={styles.legendLabel}>Warning</span>
            <span className={`status-dot red`} />
            <span className={styles.legendLabel}>Critical</span>
          </div>
        </div>

        <div className={styles.statsOverlay}>
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
        </div>
      </div>

    </div>
  )
}
