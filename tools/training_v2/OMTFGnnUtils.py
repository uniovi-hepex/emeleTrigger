# OMTFGnnUtils.py

import logging
import os
import random
import numpy as np
import torch

def setup_logging(logs_dir):
    """
    Configures logging to output to both a log file and the console.

    Parameters:
    - logs_dir (str): Directory to save log files.
    """
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, 'workflow.log'),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    # Add console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

    Parameters:
    - seed (int): Seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")
