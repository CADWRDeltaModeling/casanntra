VERSION = "residual_V1_MSTAGE_suisun"
CONFIG = "transfer_config_multistage.yml"
STEPS = ["dsm2_base", "dsm2.schism", "base.suisun"]

GRID = {
    "layers": [[{"type": "GRU", "units": 38, "trainable": True, "name": "lay1", "return_sequences": True},
                {"type": "GRU", "units": 19, "trainable": True, "name": "lay2", "return_sequences": False}]],
    "freeze": [[0, 0, 1]],
    "ndays": [105],
    "schedule": [{"dsm2_base": (0.008, 0.001, 10, 35), "dsm2.schism": (0.003, 0.001, 10, 35), "base.suisun": (0.001, 0.0005, 10, 35)}],
    "contrast_weight": [1.0, 10.0, 100.0]}
