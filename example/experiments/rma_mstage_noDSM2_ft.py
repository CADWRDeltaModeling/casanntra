VERSION = "RMA-noDSM2_MSTAGE_ft"
CONFIG = "transfer_config_rma.yml"
STEPS = ["base.ft"]

GRID = {
    "layers": [[{"type": "GRU", "units": 38, "trainable": True, "name": "lay1", "return_sequences": True},
                {"type": "GRU", "units": 19, "trainable": True, "name": "lay2", "return_sequences": False}]],
    "freeze": [[0]],
    "ndays": [105],
    "schedule": [{"base.ft": (0.008, 0.003, 30, 120)}, {"base.ft": (0.008, 0.001, 30, 120)}],
    "transfer_type": ["direct"]}
