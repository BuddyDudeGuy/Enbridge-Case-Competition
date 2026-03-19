"""
Thermal subsystem configuration for wind turbine anomaly detection.

Defines 6 thermal subsystems with sensor mappings per farm, importance weights
for TDI computation, and helper functions for downstream phases.

Generated from Phase 1.4 harmonization map.
"""

# ---------------------------------------------------------------------------
# Subsystem definitions — descriptions and TDI weights (must sum to 1.0)
# ---------------------------------------------------------------------------
THERMAL_SUBSYSTEMS = {
    "gearbox": {
        "name": "Gearbox",
        "description": "Gearbox oil and bearing temperatures — highest failure cost",
        "weight": 0.25,
    },
    "generator_bearings": {
        "name": "Generator/Bearings",
        "description": "Generator winding and bearing temperatures",
        "weight": 0.20,
    },
    "transformer": {
        "name": "Transformer",
        "description": "Transformer oil and winding temperatures",
        "weight": 0.20,
    },
    "hydraulic": {
        "name": "Hydraulic",
        "description": "Hydraulic oil temperatures (pitch/yaw systems)",
        "weight": 0.15,
    },
    "cooling": {
        "name": "Cooling",
        "description": "Cooling system fluid and component temperatures",
        "weight": 0.10,
    },
    "nacelle_ambient": {
        "name": "Nacelle/Ambient",
        "description": "Nacelle interior and ambient reference temperatures",
        "weight": 0.10,
    },
}

# ---------------------------------------------------------------------------
# Sensor mappings: farm -> subsystem -> list of _avg sensor columns
# Pulled directly from data/processed/thermal_harmonization_map.json
# ---------------------------------------------------------------------------
SUBSYSTEM_SENSORS = {
    "farm_a": {
        "gearbox": [
            "sensor_11_avg",
            "sensor_12_avg",
        ],
        "generator_bearings": [
            "sensor_13_avg",
            "sensor_14_avg",
            "sensor_15_avg",
            "sensor_16_avg",
            "sensor_17_avg",
        ],
        "transformer": [
            "sensor_38_avg",
            "sensor_39_avg",
            "sensor_40_avg",
        ],
        "hydraulic": [
            "sensor_41_avg",
        ],
        "cooling": [
            "sensor_8_avg",
            "sensor_9_avg",
            "sensor_10_avg",
        ],
        "nacelle_ambient": [
            "sensor_0_avg",
            "sensor_6_avg",
            "sensor_7_avg",
            "sensor_43_avg",
            "sensor_53_avg",
        ],
    },
    "farm_b": {
        "gearbox": [
            "sensor_34_avg",
            "sensor_35_avg",
            "sensor_36_avg",
            "sensor_37_avg",
            "sensor_38_avg",
            "sensor_39_avg",
        ],
        "generator_bearings": [
            "sensor_32_avg",
            "sensor_33_avg",
            "sensor_51_avg",
            "sensor_52_avg",
            "sensor_53_avg",
        ],
        "transformer": [
            "sensor_31_avg",
            "sensor_40_avg",
            "sensor_41_avg",
            "sensor_42_avg",
            "sensor_43_avg",
            "sensor_44_avg",
            "sensor_45_avg",
            "sensor_46_avg",
            "sensor_47_avg",
        ],
        "hydraulic": [],
        "cooling": [],
        "nacelle_ambient": [
            "sensor_8_avg",
            "sensor_22_avg",
        ],
    },
    "farm_c": {
        "gearbox": [
            "sensor_151_avg",
            "sensor_152_avg",
            "sensor_153_avg",
            "sensor_154_avg",
            "sensor_155_avg",
            "sensor_156_avg",
            "sensor_157_avg",
            "sensor_158_avg",
            "sensor_159_avg",
            "sensor_160_avg",
            "sensor_161_avg",
            "sensor_162_avg",
            "sensor_163_avg",
            "sensor_164_avg",
            "sensor_165_avg",
            "sensor_166_avg",
            "sensor_186_avg",
            "sensor_187_avg",
            "sensor_189_avg",
            "sensor_190_avg",
        ],
        "generator_bearings": [
            "sensor_18_avg",
            "sensor_19_avg",
            "sensor_20_avg",
            "sensor_21_avg",
            "sensor_168_avg",
            "sensor_169_avg",
            "sensor_173_avg",
            "sensor_174_avg",
            "sensor_194_avg",
            "sensor_195_avg",
            "sensor_196_avg",
            "sensor_197_avg",
            "sensor_198_avg",
            "sensor_199_avg",
            "sensor_200_avg",
            "sensor_201_avg",
            "sensor_202_avg",
            "sensor_203_avg",
            "sensor_204_avg",
        ],
        "transformer": [
            "sensor_167_avg",
            "sensor_188_avg",
            "sensor_191_avg",
            "sensor_192_avg",
        ],
        "hydraulic": [
            "sensor_178_avg",
            "sensor_179_avg",
        ],
        "cooling": [
            "sensor_46_avg",
            "sensor_175_avg",
            "sensor_176_avg",
            "sensor_208_avg",
            "sensor_209_avg",
            "sensor_228_avg",
            "sensor_229_avg",
            "sensor_233_avg",
            "sensor_234_avg",
        ],
        "nacelle_ambient": [
            "sensor_7_avg",
            "sensor_41_avg",
            "sensor_65_avg",
            "sensor_177_avg",
        ],
    },
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_sensors(farm: str, subsystem: str) -> list[str]:
    """Return sensor column names for a given farm and subsystem.

    Parameters
    ----------
    farm : str
        One of 'farm_a', 'farm_b', 'farm_c'.
    subsystem : str
        One of the keys in THERMAL_SUBSYSTEMS (e.g. 'gearbox').

    Returns
    -------
    list[str]
        Sensor column names (the _avg variants).
    """
    return SUBSYSTEM_SENSORS[farm][subsystem]


def get_all_thermal_sensors(farm: str) -> list[str]:
    """Return ALL thermal sensor column names for a given farm.

    Parameters
    ----------
    farm : str
        One of 'farm_a', 'farm_b', 'farm_c'.

    Returns
    -------
    list[str]
        Combined list of sensor columns across all subsystems.
    """
    sensors = []
    for subsystem in THERMAL_SUBSYSTEMS:
        sensors.extend(SUBSYSTEM_SENSORS[farm][subsystem])
    return sensors


def get_subsystem_weights() -> dict[str, float]:
    """Return a dict mapping subsystem key -> TDI weight.

    Returns
    -------
    dict[str, float]
        e.g. {'gearbox': 0.25, 'generator_bearings': 0.20, ...}
    """
    return {key: info["weight"] for key, info in THERMAL_SUBSYSTEMS.items()}
