"""
python scripts/check_arch.py path/to/scenario.yaml

Prints summary of layer names, units, trainable flag.
"""

import sys, yaml, tensorflow as tf
from casanntra.multi_stage_model_builder import MultiStageModelBuilder

INPUTS  = ["i1", "i2"]
OUTPUTS = {"o": 1.0}

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

mb = MultiStageModelBuilder(INPUTS, OUTPUTS)
mb.set_builder_args(cfg)
mb.prepro_layers = lambda self, x, y=None: list(x.values())

ins = {n: tf.keras.Input(shape=(90,), name=n) for n in INPUTS}
model = mb.build_model(ins, None)

model.summary()
for l in model.layers:
    if l.name.startswith("feature"):
        print(f"{l.name:15s} units={getattr(l,'units',None)} trainable={l.trainable}")