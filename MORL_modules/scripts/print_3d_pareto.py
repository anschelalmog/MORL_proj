import os
import sys
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'MORL_modules/run_algos'))
from hyper_morl_utils import aditional_analysis_from_save_results_3d_hv

def main():
    # You can modify this path as needed

    try:
        file_path = "/home/rotem.shezaf/MORL_proj/MORL_modules/results/Hyper-MORL/4-objdefault/0/final"
        time, hv_3d = aditional_analysis_from_save_results_3d_hv(file_path)


    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()


