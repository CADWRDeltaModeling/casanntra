VERSION = "MSCEN_RMA_v14_branch_ctrscale"
CONFIG = "transfer_config_multiscenario_rma.yml"
STEPS = ["dsm2_base", "base.multi"]

CONTRAST_SCALES = {
    "x2": 1.0, "mrz": 300.0, "pct": 1420.0, "mal": 243.0, "god": 578.0,
    "vol": 598.0, "gzl": 725.0, "bdl": 780.0, "nsl2": 898.0, "cse": 164.0,
    "emm2": 63.5, "tms": 42.6, "anh": 81.6, "jer": 35.2, "sal": 12.0,
    "frk": 25.0, "srv": 17.5, "bac": 17.8, "rsl": 16.2, "oh4": 14.3}

GRID = {
    "layers": [[{"type": "GRU", "units": 32, "return_sequences": True, "name": "lay1", "trainable": True},
                {"type": "GRU", "units": 32, "return_sequences": True, "name": "lay2", "trainable": True}]],
    "use_contrast_scales": [True],
    "repeat": [1, 2],
    "freeze": [[0, 0]],
    "ndays": [105],
    "schedule": [{"dsm2_base": (0.008, 0.001, 10, 35), "base.multi": (0.001, 0.0005, 10, 105)}],
    "source_weight": [1.0],
    "target_weight": [1.0],
    "contrast_weight": [0.5, 1],
    "per_scenario_branch": [True],
    "branch_layers": [[{"type": "GRU", "units": 32, "name": "branch_1", "trainable": True}]]}
