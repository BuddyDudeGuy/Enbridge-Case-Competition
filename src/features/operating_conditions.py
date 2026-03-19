"""
Operating condition features for Normal Behavior Models (NBMs).

These features describe the turbine's current operating state and serve as
inputs to LightGBM regressors that predict expected temperature. The NBM
learns: "given these operating conditions, what should this temperature be?"

Categories:
  - Wind speed: primary driver of power and thermal load
  - Active power: electrical output, proportional to mechanical/thermal stress
  - Ambient temperature: baseline environmental heat sink
  - Rotor/generator speed: rotational speed affects friction and heat generation
  - Pitch angle: blade pitch modulates aerodynamic load and power extraction
"""

# ---------------------------------------------------------------------------
# Per-farm operating condition features (column names in dataset CSVs)
# ---------------------------------------------------------------------------

OPERATING_FEATURES = {
    "Wind Farm A": [
        # Wind speed (m/s) — primary energy source
        "wind_speed_3_avg",   # Windspeed (measured)
        "wind_speed_4_avg",   # Estimated windspeed

        # Active power (kW)
        "power_29_avg",       # Possible grid active power
        "power_30_avg",       # Grid power

        # Ambient temperature (°C) — environmental baseline, NOT a target
        "sensor_0_avg",       # Ambient temperature

        # Rotational speeds (rpm)
        "sensor_18_avg",      # Generator rpm
        "sensor_52_avg",      # Rotor rpm

        # Pitch angle (°)
        "sensor_5_avg",       # Pitch angle
    ],

    "Wind Farm B": [
        # Wind speed (m/s)
        "wind_speed_59_avg",  # Wind speed 1
        "wind_speed_60_avg",  # Wind speed 2
        "wind_speed_61_avg",  # Wind speed (combined/primary)

        # Active power (kW)
        "power_62_avg",       # Active power
        "power_58_avg",       # Available power

        # Ambient/outside temperature (°C)
        "sensor_8_avg",       # Outside temperature

        # Rotational speeds (rpm)
        "sensor_19_avg",      # Generator converter rotational speed
        "sensor_20_avg",      # Gearbox rotational speed
        "sensor_25_avg",      # Rotor speed

        # Pitch angle (°)
        "sensor_10_avg",      # Pitch angle
    ],

    "Wind Farm C": [
        # Wind speed (m/s)
        "wind_speed_235_avg", # Wind speed 1
        "wind_speed_236_avg", # Wind speed 1+2 (averaged)
        "wind_speed_237_avg", # Wind speed 2

        # Active power (kW)
        "power_6_avg",        # Active power HV grid
        "power_5_avg",        # Active power grid side converter

        # Ambient temperature (°C)
        "sensor_7_avg",       # Ambient temperature

        # Rotational speeds (1/min and rad/s)
        "sensor_144_avg",     # Rotor speed 1 (1/min)
        "sensor_145_avg",     # Rotor speed 2 (1/min)
        "sensor_8_avg",       # Generator angle speed (rad/s)

        # Pitch angle (°)
        "sensor_76_avg",      # Min pitch angle
        "sensor_103_avg",     # Position rotor blade axis 1
        "sensor_104_avg",     # Position rotor blade axis 2
        "sensor_105_avg",     # Position rotor blade axis 3
    ],
}

# ---------------------------------------------------------------------------
# Feature metadata: human-readable descriptions + units per column
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS = {
    "Wind Farm A": {
        "wind_speed_3_avg":  ("Windspeed", "m/s"),
        "wind_speed_4_avg":  ("Estimated windspeed", "m/s"),
        "power_29_avg":      ("Possible grid active power", "kW"),
        "power_30_avg":      ("Grid power", "kW"),
        "sensor_0_avg":      ("Ambient temperature", "°C"),
        "sensor_18_avg":     ("Generator rpm", "rpm"),
        "sensor_52_avg":     ("Rotor rpm", "rpm"),
        "sensor_5_avg":      ("Pitch angle", "°"),
    },
    "Wind Farm B": {
        "wind_speed_59_avg": ("Wind speed 1", "m/s"),
        "wind_speed_60_avg": ("Wind speed 2", "m/s"),
        "wind_speed_61_avg": ("Wind speed", "m/s"),
        "power_62_avg":      ("Active power", "kW"),
        "power_58_avg":      ("Available power", "kW"),
        "sensor_8_avg":      ("Outside temperature", "°C"),
        "sensor_19_avg":     ("Generator converter rotational speed", "rpm"),
        "sensor_20_avg":     ("Gearbox rotational speed", "rpm"),
        "sensor_25_avg":     ("Rotor speed", "rpm"),
        "sensor_10_avg":     ("Pitch angle", "°"),
    },
    "Wind Farm C": {
        "wind_speed_235_avg": ("Wind speed 1", "m/s"),
        "wind_speed_236_avg": ("Wind speed 1+2", "m/s"),
        "wind_speed_237_avg": ("Wind speed 2", "m/s"),
        "power_6_avg":        ("Active power HV grid", "kW"),
        "power_5_avg":        ("Active power grid side converter", "kW"),
        "sensor_7_avg":       ("Ambient temperature", "°C"),
        "sensor_144_avg":     ("Rotor speed 1", "1/min"),
        "sensor_145_avg":     ("Rotor speed 2", "1/min"),
        "sensor_8_avg":       ("Generator angle speed", "rad/s"),
        "sensor_76_avg":      ("Min pitch angle", "°"),
        "sensor_103_avg":     ("Position rotor blade axis 1", "°"),
        "sensor_104_avg":     ("Position rotor blade axis 2", "°"),
        "sensor_105_avg":     ("Position rotor blade axis 3", "°"),
    },
}

# ---------------------------------------------------------------------------
# Categorical grouping: which category each feature belongs to
# ---------------------------------------------------------------------------

FEATURE_CATEGORIES = {
    "Wind Farm A": {
        "wind_speed":    ["wind_speed_3_avg", "wind_speed_4_avg"],
        "active_power":  ["power_29_avg", "power_30_avg"],
        "ambient_temp":  ["sensor_0_avg"],
        "rotor_speed":   ["sensor_52_avg"],
        "generator_speed": ["sensor_18_avg"],
        "pitch_angle":   ["sensor_5_avg"],
    },
    "Wind Farm B": {
        "wind_speed":    ["wind_speed_59_avg", "wind_speed_60_avg", "wind_speed_61_avg"],
        "active_power":  ["power_62_avg", "power_58_avg"],
        "ambient_temp":  ["sensor_8_avg"],
        "rotor_speed":   ["sensor_25_avg"],
        "generator_speed": ["sensor_19_avg"],
        "gearbox_speed": ["sensor_20_avg"],
        "pitch_angle":   ["sensor_10_avg"],
    },
    "Wind Farm C": {
        "wind_speed":    ["wind_speed_235_avg", "wind_speed_236_avg", "wind_speed_237_avg"],
        "active_power":  ["power_6_avg", "power_5_avg"],
        "ambient_temp":  ["sensor_7_avg"],
        "rotor_speed":   ["sensor_144_avg", "sensor_145_avg"],
        "generator_speed": ["sensor_8_avg"],
        "pitch_angle":   ["sensor_76_avg", "sensor_103_avg", "sensor_104_avg", "sensor_105_avg"],
    },
}


def get_operating_features(farm: str) -> list[str]:
    """Return the list of operating condition column names for a given farm.

    Parameters
    ----------
    farm : str
        One of "Wind Farm A", "Wind Farm B", "Wind Farm C".

    Returns
    -------
    list[str]
        Column names to use as NBM inputs.

    Raises
    ------
    KeyError
        If the farm name is not recognized.
    """
    if farm not in OPERATING_FEATURES:
        raise KeyError(
            f"Unknown farm '{farm}'. Choose from: {list(OPERATING_FEATURES.keys())}"
        )
    return OPERATING_FEATURES[farm]


def get_feature_descriptions(farm: str) -> dict[str, tuple[str, str]]:
    """Return {column: (description, unit)} for a farm's operating features."""
    if farm not in FEATURE_DESCRIPTIONS:
        raise KeyError(
            f"Unknown farm '{farm}'. Choose from: {list(FEATURE_DESCRIPTIONS.keys())}"
        )
    return FEATURE_DESCRIPTIONS[farm]


def get_feature_categories(farm: str) -> dict[str, list[str]]:
    """Return {category: [columns]} for a farm's operating features."""
    if farm not in FEATURE_CATEGORIES:
        raise KeyError(
            f"Unknown farm '{farm}'. Choose from: {list(FEATURE_CATEGORIES.keys())}"
        )
    return FEATURE_CATEGORIES[farm]
