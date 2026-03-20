"""
Normal Behavior Model (NBM) configuration for wind turbine anomaly detection.

Defines which input features (operating conditions) and target sensors
(temperatures to predict) are used for each farm's LightGBM regressors.

Each NBM learns: "given these operating conditions, what SHOULD this
temperature sensor read?" Residuals between predicted and actual indicate
potential degradation.

Target selection rationale:
  - Gearbox: oil temperature — oil integrates heat from all gearbox components,
    making it the best single-sensor proxy for overall gearbox thermal health.
  - Generator/Bearings: drive-end bearing temp — closest to the coupling and
    most common failure point for bearing degradation.
  - Transformer: core/oil temperature — most sensitive to internal overheating
    and insulation degradation.
  - Hydraulic: oil tank temperature — integrates heat from pump, valves, and
    actuators in the pitch/yaw hydraulic circuit.
  - Cooling: main cooling water temperature — reflects overall cooling system
    performance and heat rejection capacity.
  - Nacelle/Ambient: SKIPPED — ambient temperature is an INPUT (environmental
    baseline), not something the turbine's behavior determines.
"""

from src.features.operating_conditions import OPERATING_FEATURES

# ---------------------------------------------------------------------------
# NBM target sensors: farm -> subsystem -> target column name
# One primary target per subsystem per farm (the _avg variant).
# ---------------------------------------------------------------------------

NBM_TARGETS = {
    "Wind Farm A": {
        "gearbox": {
            "column": "sensor_12_avg",
            "description": "Temperature oil in gearbox",
            "unit": "°C",
            "rationale": "Oil integrates heat from all gearbox components",
        },
        "generator_bearings": {
            "column": "sensor_13_avg",
            "description": "Temperature in generator bearing 2 (Drive End)",
            "unit": "°C",
            "rationale": "Drive-end bearing is closest to failure point",
        },
        "transformer": {
            "column": "sensor_38_avg",
            "description": "Temperature in HV transformer phase L1",
            "unit": "°C",
            "rationale": "Phase winding temp most sensitive to overheating",
        },
        "hydraulic": {
            "column": "sensor_41_avg",
            "description": "Temperature oil in hydraulic group",
            "unit": "°C",
            "rationale": "Only hydraulic temp sensor — integrates system heat",
        },
        "cooling": {
            "column": "sensor_10_avg",
            "description": "Temperature in the VCS cooling water",
            "unit": "°C",
            "rationale": "Main cooling water reflects heat rejection capacity",
        },
    },

    "Wind Farm B": {
        "gearbox": {
            "column": "sensor_39_avg",
            "description": "Gearbox oil tank temperature",
            "unit": "°C",
            "rationale": "Oil tank temp integrates heat from all gearbox components",
        },
        "generator_bearings": {
            "column": "sensor_32_avg",
            "description": "Generator bearing temperature 1",
            "unit": "°C",
            "rationale": "Primary generator bearing — drive-end failure indicator",
        },
        "transformer": {
            "column": "sensor_40_avg",
            "description": "Transformator core temperature",
            "unit": "°C",
            "rationale": "Core temp is most sensitive to transformer overheating",
        },
        # hydraulic: no sensors available for Farm B — omitted
        # cooling: no sensors available for Farm B — omitted
    },

    "Wind Farm C": {
        "gearbox": {
            "column": "sensor_186_avg",
            "description": "Gearbox oil temperature 1",
            "unit": "°C",
            "rationale": "Oil temp integrates heat from all gearbox components",
        },
        "generator_bearings": {
            "column": "sensor_196_avg",
            "description": "Rotor bearing temperature 1",
            "unit": "°C",
            "rationale": "Primary rotor bearing — drive-end degradation indicator",
        },
        "transformer": {
            "column": "sensor_191_avg",
            "description": "Oil temperature 1 main transformer",
            "unit": "°C",
            "rationale": "Main transformer oil temp — most sensitive to overheating",
        },
        "hydraulic": {
            "column": "sensor_178_avg",
            "description": "Hydraulic oil tank temperature 1",
            "unit": "°C",
            "rationale": "Oil tank integrates heat from pitch/yaw hydraulic system",
        },
        "cooling": {
            "column": "sensor_228_avg",
            "description": "Cooling water temp. generator inlet 1",
            "unit": "°C",
            "rationale": "Main cooling water inlet reflects cooling system performance",
        },
    },
}

# ---------------------------------------------------------------------------
# NBM inputs: reuse operating conditions from Phase 1.6
# ---------------------------------------------------------------------------

NBM_INPUTS = OPERATING_FEATURES


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_nbm_config(farm: str) -> dict:
    """Return the full NBM configuration for a given farm.

    Parameters
    ----------
    farm : str
        One of "Wind Farm A", "Wind Farm B", "Wind Farm C".

    Returns
    -------
    dict
        {
            'inputs': list[str]  — input feature column names,
            'targets': dict[str, str] — {subsystem: target_column_name}
        }

    Raises
    ------
    KeyError
        If the farm name is not recognized.
    """
    if farm not in NBM_TARGETS:
        raise KeyError(
            f"Unknown farm '{farm}'. Choose from: {list(NBM_TARGETS.keys())}"
        )

    targets = {
        subsystem: info["column"]
        for subsystem, info in NBM_TARGETS[farm].items()
    }

    return {
        "inputs": NBM_INPUTS[farm],
        "targets": targets,
    }


def get_all_targets(farm: str) -> list[str]:
    """Return a flat list of all target sensor column names for a farm.

    Parameters
    ----------
    farm : str
        One of "Wind Farm A", "Wind Farm B", "Wind Farm C".

    Returns
    -------
    list[str]
        Target column names across all subsystems.

    Raises
    ------
    KeyError
        If the farm name is not recognized.
    """
    if farm not in NBM_TARGETS:
        raise KeyError(
            f"Unknown farm '{farm}'. Choose from: {list(NBM_TARGETS.keys())}"
        )

    return [info["column"] for info in NBM_TARGETS[farm].values()]


def get_target_info(farm: str, subsystem: str) -> dict:
    """Return full metadata for a specific target sensor.

    Parameters
    ----------
    farm : str
        One of "Wind Farm A", "Wind Farm B", "Wind Farm C".
    subsystem : str
        One of the subsystem keys (e.g. 'gearbox', 'generator_bearings').

    Returns
    -------
    dict
        {'column': str, 'description': str, 'unit': str, 'rationale': str}

    Raises
    ------
    KeyError
        If the farm or subsystem is not recognized.
    """
    if farm not in NBM_TARGETS:
        raise KeyError(
            f"Unknown farm '{farm}'. Choose from: {list(NBM_TARGETS.keys())}"
        )
    if subsystem not in NBM_TARGETS[farm]:
        available = list(NBM_TARGETS[farm].keys())
        raise KeyError(
            f"Subsystem '{subsystem}' not available for {farm}. "
            f"Available: {available}"
        )

    return NBM_TARGETS[farm][subsystem]


def get_model_count(farm: str) -> int:
    """Return the number of NBMs to train for a given farm.

    Parameters
    ----------
    farm : str
        One of "Wind Farm A", "Wind Farm B", "Wind Farm C".

    Returns
    -------
    int
        Number of target sensors (= number of models).
    """
    return len(NBM_TARGETS[farm])
