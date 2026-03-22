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
  },
  {
    id: 'B',
    name: 'Farm B',
    location: 'North Sea (Offshore)',
    lat: 54.0,
    lng: 7.0,
    turbines: 9,
    avgTdi: 19.45,
    health: 'green' as const,
    windSpeed: 11.8,
    powerOutput: 38.7,
  },
  {
    id: 'C',
    name: 'Farm C',
    location: 'North Sea (Offshore)',
    lat: 54.2,
    lng: 7.3,
    turbines: 22,
    avgTdi: 60.94,
    health: 'red' as const,
    windSpeed: 10.5,
    powerOutput: 84.2,
  },
]

export const fleetKpis = {
  totalTurbines: 36,
  activeAlerts: 4,
  fleetHealth: 87,
  uptime: 96.4,
  careScore: 0.61,
  totalOutput: 135.3,
  avgWindSpeed: 9.8,
}

export const subsystems = [
  { name: 'Gearbox', health: 82, weight: 0.25 },
  { name: 'Generator', health: 91, weight: 0.20 },
  { name: 'Transformer', health: 76, weight: 0.20 },
  { name: 'Hydraulic', health: 88, weight: 0.15 },
  { name: 'Cooling', health: 94, weight: 0.10 },
  { name: 'Nacelle', health: 97, weight: 0.10 },
]

export const alerts = [
  {
    id: 1,
    turbine: 'T-14',
    farm: 'C',
    subsystem: 'Gearbox',
    message: 'Bearing temp 12°C above NBM prediction',
    severity: 'red' as const,
    time: '2 min ago',
  },
  {
    id: 2,
    turbine: 'T-07',
    farm: 'C',
    subsystem: 'Transformer',
    message: 'Core temp rising — CUSUM threshold breached',
    severity: 'red' as const,
    time: '8 min ago',
  },
  {
    id: 3,
    turbine: 'T-19',
    farm: 'C',
    subsystem: 'Hydraulic',
    message: 'Oil temp anomaly detected via EWMA',
    severity: 'amber' as const,
    time: '14 min ago',
  },
  {
    id: 4,
    turbine: 'T-03',
    farm: 'A',
    subsystem: 'Generator',
    message: 'Bearing vibration slightly elevated',
    severity: 'amber' as const,
    time: '27 min ago',
  },
  {
    id: 5,
    turbine: 'T-22',
    farm: 'C',
    subsystem: 'Cooling',
    message: 'Water inlet temp 3°C above baseline',
    severity: 'amber' as const,
    time: '41 min ago',
  },
  {
    id: 6,
    turbine: 'T-05',
    farm: 'B',
    subsystem: 'Gearbox',
    message: 'Oil temp within normal range — resolved',
    severity: 'green' as const,
    time: '1h ago',
  },
]

export const powerHistory = [
  { hour: '06:00', mw: 98 },
  { hour: '07:00', mw: 112 },
  { hour: '08:00', mw: 125 },
  { hour: '09:00', mw: 131 },
  { hour: '10:00', mw: 128 },
  { hour: '11:00', mw: 135 },
  { hour: '12:00', mw: 142 },
  { hour: '13:00', mw: 138 },
  { hour: '14:00', mw: 145 },
  { hour: '15:00', mw: 139 },
  { hour: '16:00', mw: 132 },
  { hour: '17:00', mw: 118 },
]

export const chatConversation = [
  {
    role: 'user' as const,
    text: 'Which turbines need immediate attention?',
  },
  {
    role: 'ai' as const,
    text: 'Based on current TDI scores, 2 turbines are critical on Farm C:\n\n• T-14 — Gearbox bearing temp is 12°C above the Normal Behavior Model prediction. TDI score: 84. Pattern matches historical gearbox failure Event 72 from Farm A (91% similarity). Recommend inspection within 48 hours.\n\n• T-07 — Transformer core temp has breached the CUSUM threshold. TDI score: 71. Rising trend started 6 hours ago. Recommend prioritizing due to offshore access constraints.',
  },
]

export const suggestedQueries = [
  'Farm C health trends',
  'What caused T-14 alert?',
  'Compare farms A vs B',
  'Next maintenance window',
  'Show CARE score breakdown',
  'Turbine health scores',
]
