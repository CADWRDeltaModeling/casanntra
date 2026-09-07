VERSION = "SMOKE_TEST"
CONFIG = "transfer_config_multiscenario_rma.yml"
STEPS = ["dsm2_base", "base.multi"]

GRID = {
    "layers": [[{"type": "GRU", "units": 8, "return_sequences": True, "name": "lay1", "trainable": True},
                {"type": "GRU", "units": 4, "return_sequences": False, "name": "lay2", "trainable": True}]],
    "freeze": [[0, 1]],
    "ndays": [30],
    "schedule": [{"dsm2_base": (0.008, 0.001, 1, 1), "base.multi": (0.001, 0.0005, 1, 1)}],
    "contrast_weight": [1.0]}
