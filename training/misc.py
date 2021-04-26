import glob
import os
import re

from pathlib import Path

def get_parent_dir(run_dir):
    out_dir = Path(run_dir).parent

    return out_dir

def locate_latest_pkl(out_dir):
    all_pickle_names = sorted(glob.glob(os.path.join(out_dir, '0*', 'network-*.pkl')))

    try:
        latest_pickle_name = all_pickle_names[-1]
    except IndexError:
        latest_pickle_name = None

    return latest_pickle_name

def parse_kimg_from_network_name(network_pickle_name):

    if network_pickle_name is not None:
        resume_run_id = os.path.basename(os.path.dirname(network_pickle_name))
        RE_KIMG = re.compile('network-snapshot-(\d+).pkl')
        try:
            kimg = int(RE_KIMG.match(os.path.basename(network_pickle_name)).group(1))
        except AttributeError:
            kimg = 0.0
    else:
        kimg = 0.0

    return float(kimg)


def parse_augment_p_from_log(network_pickle_name):

    if network_pickle_name is not None:
        network_folder_name = os.path.dirname(network_pickle_name)
        log_file_name = network_folder_name + "/log.txt"

        try:
            with open(log_file_name, "r") as f:
                # Tokenize each line starting with the word 'tick'
                lines = [
                    l.strip().split() for l in f.readlines() if l.startswith("tick")
                ]
        except FileNotFoundError:
            lines = []

        # Extract the last token of each line for which the second to last token is 'augment'
        values = [
            tokens[-1]
            for tokens in lines
            if len(tokens) > 1 and tokens[-2] == "augment"
        ]

        if len(values)>0:
            augment_p = float(values[-1])
        else:
            augment_p = 0.0
    else:
        augment_p = 0.0

    return float(augment_p)
