import sys
import os
import time
import tempfile
import csv
import json
import sys
import numpy as np
from typing import Dict, Any, List, Tuple
import pdb
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'MORL_modules'))
#sys.path.append(os.path.join(project_root, 'energy-net'))




from utils.utils import moving_average, plot_results_scalarized
from utils.callbacks import SaveOnBestTrainingRewardCallback

def main():
    plot_results_scalarized("MORL_modules/logs/mosac_monitor_old/", title="Learning Curve")

if __name__ == "__main__":
    main()