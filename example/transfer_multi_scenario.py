"""
transfer_multi_scenarios.py

Run the staged transfer pipeline with a *single* multi-scenario step
that trains SCHISM-Base + N scenario heads (contrastive) in one pass.

Config file: transfer_config_multi.yaml
"""

from casanntra.staged_learning import process_config

def main():
    configfile = "transfer_config_multi.yaml"
    steps_to_run = ["dsm2_base", "dsm2.schism", "base.multi"]
    process_config(configfile, steps_to_run)

if __name__ == "__main__":
    main()