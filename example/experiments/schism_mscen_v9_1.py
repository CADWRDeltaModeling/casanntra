VERSION = "MSCEN_v9.1"
CONFIG = "transfer_config_multiscenario.yml"
STEPS = ["dsm2.schism", "base.multi"]

GRID = {
    "layers": [[{"type": "GRU", "units": 38, "return_sequences": True, "name": "lay1", "trainable": True},
                {"type": "GRU", "units": 19, "return_sequences": False, "name": "lay2", "trainable": True}],
               [{"type": "GRU", "units": 32, "return_sequences": True, "name": "lay1", "trainable": True},
                {"type": "GRU", "units": 16, "return_sequences": False, "name": "lay2", "trainable": True}]],
    "freeze": [[0, 1], [0, 2]],
    "ndays": [105],
    "schedule": [{"dsm2.schism": (0.003, 0.001, 10, 35), "base.multi": (0.001, 0.0005, 10, 35)}],
    "source_weight": [1.0],
    "target_weight": [1.0],
    "contrast_weight": [0.5],
    "per_scenario_branch": [False],
    "branch_layers": [[]]}
