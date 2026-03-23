'use client'

import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceLine,
  ReferenceArea,
  ResponsiveContainer,
} from 'recharts'
import styles from './PredictiveChart.module.css'

interface EventChartPoint {
  day: number
  actual: number
  predicted: number
  residual: number
}

interface PredictiveChartProps {
  title: string
  subtitle: string
  subsystem: string
  r2: number
  warningDays: number
  totalDays: number
  data: EventChartPoint[]
  faultStart: number
  height?: number
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload || !payload.length) return null

  const predicted = payload.find((p: any) => p.dataKey === 'predicted')
  const actual = payload.find((p: any) => p.dataKey === 'actual')
  const residual = actual && predicted ? actual.value - predicted.value : 0

  return (
    <div className={styles.tooltip}>
      <span className={styles.tooltipDay}>Day {label}</span>
      <span className={styles.tooltipPredicted}>
        Expected: {predicted?.value?.toFixed(1)}&deg;C
      </span>
      <span className={styles.tooltipActual}>
        Actual: {actual?.value?.toFixed(1)}&deg;C
      </span>
      <span className={styles.tooltipResidual}>
        Residual: {residual >= 0 ? '+' : ''}{residual.toFixed(1)}&deg;C
      </span>
    </div>
  )
}

export default function PredictiveChart({
  title,
  subtitle,
  subsystem,
  r2,
  warningDays,
  totalDays,
  data,
  faultStart,
  height = 400,
}: PredictiveChartProps) {
  const warningStart = faultStart - warningDays
  const warningCoversAll = warningDays >= totalDays * 0.8

  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <div>
          <h3 className={styles.title}>{title}</h3>
          <span className={styles.subtitle}>{subtitle}</span>
        </div>
        <span className={styles.r2Badge}>R&sup2; = {r2.toFixed(3)}</span>
      </div>

      <div className={styles.warningBanner}>
        {warningCoversAll
          ? `${warningDays}-day anomaly detected from Day 1 — our model catches degradation across the entire prediction window`
          : `${warningDays}-day early warning before fault onset`}
      </div>

      <div className={styles.chartWrapper}>
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={data} margin={{ top: 20, right: 30, bottom: 20, left: 10 }}>
            <defs>
              <linearGradient id="residualGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" stopOpacity={0.25} />
                <stop offset="100%" stopColor="#22c55e" stopOpacity={0.05} />
              </linearGradient>
            </defs>

            <XAxis
              dataKey="day"
              tick={{ fontSize: 11, fontFamily: 'var(--font-mono), monospace', fill: 'rgba(255,255,255,0.5)' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              tickLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              interval={Math.max(0, Math.floor(data.length / 6) - 1)}
              tickFormatter={(val: number) => Math.round(val).toString()}
            />
            <YAxis
              tick={{ fontSize: 11, fontFamily: 'var(--font-mono), monospace', fill: 'rgba(255,255,255,0.5)' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              tickLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              label={{
                value: 'Temperature (\u00B0C)',
                angle: -90,
                position: 'insideLeft',
                style: { fontSize: 11, fontFamily: 'var(--font-mono), monospace', fill: 'rgba(255,255,255,0.4)' },
              }}
            />

            {!warningCoversAll && (
              <ReferenceArea
                x1={warningStart}
                x2={faultStart}
                fill="rgba(245, 158, 11, 0.12)"
                stroke="rgba(245, 158, 11, 0.3)"
                strokeDasharray="4 4"
              />
            )}

            <ReferenceLine
              x={faultStart}
              stroke="#ef4444"
              strokeDasharray="6 3"
              strokeWidth={1.5}
              label={{
                value: 'Fault Window',
                position: 'insideTopRight',
                style: { fontSize: 12, fontFamily: 'var(--font-mono), monospace', fill: '#ef4444', fontWeight: 600 },
              }}
            />

            <Area
              type="monotone"
              dataKey="residual"
              fill="url(#residualGradient)"
              stroke="none"
              baseLine={0}
              isAnimationActive={false}
              legendType="none"
            />

            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="Expected Temperature"
              activeDot={{ r: 4, stroke: '#3b82f6', fill: '#1e1e26' }}
            />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#f97316"
              strokeWidth={2}
              dot={false}
              name="Actual"
              activeDot={{ r: 4, stroke: '#f97316', fill: '#1e1e26' }}
            />

            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{
                fontSize: 11,
                fontFamily: 'var(--font-mono), monospace',
                color: 'rgba(255,255,255,0.6)',
                paddingTop: 16,
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
