kindOfData = [
    "EventTier",
    "VanillaTier",
    "WeatherTier",
    "VolumeTier",
    "VolumeJCD",
]

kindOfModel = [
    "Continuity",
    "ContinuityDecoder",
    "TorchEncoder",
    "Linformer",
]

listOfExt = [
    "csv",
]

# Checkpoint name at the end of each epoch
checkpoint_name = "transformer_checkpoint"

# Inputs
inputStructure = {
    "VanillaTier": {
        "utils": "tier_utils",
    },
    "WeatherTier": {
        "utils": "tier_utils",
    },
    "EventTier": {
        "utils": "tier_utils",
    },
    "VolumeTier": {
        "utils": "tier_utils",
    },
    "VolumeJCD": {
        "utils": "JCD_utils",
    },
}

# Ouptpus
outputStructure = {
    "VanillaTier": {
        "columns": [
            "date",
            "start_station_id",
            "time_slot",
            "demand_count",
            "prediction_count",
        ],
    },
    "WeatherTier": {
        "columns": [
            "date",
            "start_station_id",
            "time_slot",
            "demand_count",
            "prediction_count",
        ],
    },
    "EventTier": {
        "columns": [
            "date",
            "start_station_id",
            "time_slot",
            "demand_count",
            "prediction_count",
        ],
    },
    "VolumeTier": {
        "columns": [
            "date",
            "start_station_id",
            "time_slot",
            "demand_count",
            "prediction_count",
        ],
    },
    "VolumeJCD": {
        "columns": [
            "date",
            "station",
            "demand_count",
            "prediction_count",
        ],
    },
}
