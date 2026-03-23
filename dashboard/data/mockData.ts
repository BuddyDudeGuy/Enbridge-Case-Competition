export const farms = [
  {
    id: 'A',
    name: 'Farm A',
    location: 'Portugal (Onshore)',
    lat: 39.5,
    lng: -8.0,
    turbines: 5,
    avgTdi: 13.25,
    health: 'green' as const,
    windSpeed: 7.2,
    powerOutput: 12.4,
    tempDeviation: 1.2,
  },
  {
    id: 'B',
    name: 'Farm B',
    location: 'North Sea (Offshore)',
    lat: 55.5,
    lng: 5.0,
    turbines: 9,
    avgTdi: 19.45,
    health: 'green' as const,
    windSpeed: 11.8,
    powerOutput: 38.7,
    tempDeviation: 2.1,
  },
  {
    id: 'C',
    name: 'Farm C',
    location: 'North Sea (Offshore)',
    lat: 52.5,
    lng: 9.0,
    turbines: 22,
    avgTdi: 60.94,
    health: 'red' as const,
    windSpeed: 10.5,
    powerOutput: 84.2,
    tempDeviation: 8.7,
  },
]

export const fleetKpis = {
  totalTurbines: 36,
  activeAlerts: 4,
  fleetTdi: 31.2,
  careScore: 0.608,
  avgTempDeviation: 4.3,
  detectionRate: 53,
  falseAlarmRate: 6,
  turbinesGreen: 28,
  turbinesYellow: 5,
  turbinesRed: 3,
}

export const subsystems = [
  { name: 'Gearbox', health: 82, weight: 0.25, avgDeviation: 5.4 },
  { name: 'Generator', health: 85, weight: 0.20, avgDeviation: 2.1 },
  { name: 'Transformer', health: 30, weight: 0.20, avgDeviation: 7.8, limitation: 'Requires electrical telemetry' },
  { name: 'Hydraulic', health: 45, weight: 0.15, avgDeviation: 3.2, limitation: 'Requires actuator telemetry' },
  { name: 'Cooling', health: 90, weight: 0.10, avgDeviation: 1.4 },
]

export const alerts = [
  {
    id: 1,
    turbine: 'T-14',
    farm: 'C',
    subsystem: 'Gearbox',
    message: 'Bearing temp 12°C above NBM prediction — TDI 84',
    severity: 'red' as const,
    time: '2 min ago',
    tdi: 84,
  },
  {
    id: 2,
    turbine: 'T-07',
    farm: 'C',
    subsystem: 'Transformer',
    message: 'Core temp rising — CUSUM threshold breached — TDI 71',
    severity: 'red' as const,
    time: '8 min ago',
    tdi: 71,
  },
  {
    id: 3,
    turbine: 'T-19',
    farm: 'C',
    subsystem: 'Hydraulic',
    message: 'Oil temp anomaly via EWMA — TDI 48',
    severity: 'amber' as const,
    time: '14 min ago',
    tdi: 48,
  },
  {
    id: 4,
    turbine: 'T-03',
    farm: 'A',
    subsystem: 'Generator',
    message: 'Stator winding temp elevated 4°C — TDI 38',
    severity: 'amber' as const,
    time: '27 min ago',
    tdi: 38,
  },
  {
    id: 5,
    turbine: 'T-22',
    farm: 'C',
    subsystem: 'Cooling',
    message: 'Water inlet temp 3°C above baseline — TDI 33',
    severity: 'amber' as const,
    time: '41 min ago',
    tdi: 33,
  },
  {
    id: 6,
    turbine: 'T-05',
    farm: 'B',
    subsystem: 'Gearbox',
    message: 'Oil temp within normal range — resolved — TDI 12',
    severity: 'green' as const,
    time: '1h ago',
    tdi: 12,
  },
]

export const tdiHistory = [
  { hour: '06:00', tdi: 22 },
  { hour: '07:00', tdi: 24 },
  { hour: '08:00', tdi: 23 },
  { hour: '09:00', tdi: 26 },
  { hour: '10:00', tdi: 28 },
  { hour: '11:00', tdi: 31 },
  { hour: '12:00', tdi: 29 },
  { hour: '13:00', tdi: 33 },
  { hour: '14:00', tdi: 35 },
  { hour: '15:00', tdi: 32 },
  { hour: '16:00', tdi: 30 },
  { hour: '17:00', tdi: 31 },
]

export const farmDetails: Record<string, {
  strongSubsystems: string[]
  keyMetrics: Record<string, number>
  figures: { src: string; caption: string }[]
  keyEvent: string
  detectionHighlight: string
  tdiScores: Record<string, number>
}> = {
  A: {
    strongSubsystems: ['Gearbox', 'Cooling', 'Generator Bearings'],
    keyMetrics: { gearboxR2: 0.804, coolingR2: 0.849, genBearingR2: 0.767 },
    tdiScores: { Gearbox: 82, Cooling: 90, 'Generator Bearings': 85 } as Record<string, number>,
    figures: [
      { src: '/figures/farm_a_gearbox_zoomed.png', caption: 'Gearbox Event 72 — 7-day early warning detected' },
      { src: '/figures/farm_a_generator_bearing_normal_vs_anomaly.png', caption: 'Generator bearing anomaly vs normal behavior' },
    ],
    keyEvent: 'Event 72: Gearbox oil temp diverged 7 days before failure',
    detectionHighlight: 'R² 0.80 Gearbox · R² 0.85 Cooling · R² 0.77 Gen Bearings',
  },
  B: {
    strongSubsystems: ['Gearbox', 'Generator Bearings'],
    keyMetrics: { gearboxR2: 0.768, genBearingR2: 0.50 },
    tdiScores: { Gearbox: 78, 'Generator Bearings': 72 } as Record<string, number>,
    figures: [
      { src: '/figures/timeline_farm_b_event53_bearing.png', caption: '42-day bearing degradation — earliest warning in fleet' },
      { src: '/figures/farm_b_bearing_degradation_normal_vs_anomaly.png', caption: 'Rotor bearing: normal vs degradation pattern' },
    ],
    keyEvent: 'Event 53: Bearing degradation visible 42 days before replacement',
    detectionHighlight: '42-Day Early Warning · R² 0.77 Gearbox',
  },
  C: {
    strongSubsystems: ['Gearbox', 'Cooling', 'Generator Bearings'],
    keyMetrics: { gearboxR2: 0.788, coolingR2: 0.761, genBearingR2: 0.721 },
    tdiScores: { Gearbox: 79, Cooling: 85, 'Generator Bearings': 76 } as Record<string, number>,
    figures: [
      { src: '/figures/farm_c_cooling_normal_vs_anomaly.png', caption: 'Cooling valve misposition — dramatic thermal spike detected' },
      { src: '/figures/farm_c_hydraulic_normal_vs_anomaly.png', caption: 'Hydraulic oil 25°C → 70°C spike in 4 minutes' },
    ],
    keyEvent: 'Event 44: Cooling valve failure caught by thermal signature',
    detectionHighlight: '63% Detection Rate · R² 0.79 Gearbox · R² 0.76 Cooling',
  },
}
