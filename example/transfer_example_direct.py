"""
transfer_example_direct.py

A minimal script that runs the direct staged transfer learning pipeline
using the config file: transfer_config_direct.yaml
"""

import os
from casanntra.staged_learning import process_config

def main():
    # If you named the config file differently, adjust here:
    configfile = "transfer_config_direct.yaml"

    # We run the three steps:
    steps_to_run = ["dsm2_base","dsm2.schism","base.suisun"]
    process_config(configfile, steps_to_run)

if __name__ == "__main__":
    main()