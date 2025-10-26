# example/transfer_multi_scenario.py

import os
from casanntra.staged_learning import process_config

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    configfile = os.path.join(here, "transfer_config_multi.yaml")
    steps_to_run = ["dsm2_base", "dsm2.schism", "base.multi"]
    os.makedirs(os.path.join(os.path.dirname(here), "output"), exist_ok=True)
    process_config(configfile, steps_to_run)

if __name__ == "__main__":
    main()