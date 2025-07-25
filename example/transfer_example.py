from casanntra.staged_learning import process_config
import multiprocessing as mp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    """Ultimately we may launch this using argparse, but it is hard wired right now.
    The first argument, configfile, gives the configuration steps for DSM2 base, DSM2->SCHISM
    and SCHISM,base -> SCHISM,suisun.

    My visualization of the cross validation output is vis_output.py.
    It is in much more rudimentary state.
    """
    configfile = "transfer_config_schism_v4.yml"

    # This selects the steps you want to run
    process_config(configfile, ["dsm2_base"])
    process_config(configfile, [ "dsm2.schism", "base.suisun","base.slr", "base.ft", "base.cache"])
    process_config(configfile, ["dsm2.rma", "rma_base.suisun", "rma_base.ft", "rma_base.cache"])


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    main()
