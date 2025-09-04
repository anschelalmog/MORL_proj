import numpy as np
#monkey patch for older version use in dependecies
if not hasattr(np, 'bool'):
    np.bool = np.bool_ # or np.bool = np.bool_

import os, sys, signal
import random

from multiprocessing import Process, Queue, current_process, freeze_support
import argparse
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

#sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'energy-net'))
sys.path.append(project_root)
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'energy-net'))

#sys.path.append(os.path.join(project_root, 'MORL_modules'))

torch.set_num_threads(1)
parser = argparse.ArgumentParser()
parser.add_argument('--num-seeds', type=int, default=1)
args = parser.parse_args()

commands = []
exp_names = ['default']
task_num = len(exp_names)
num_processes=args.num_seeds*task_num
for k in range(task_num):
    exp_name = exp_names[k]
    # Ant
    random.seed(2000)
    for i in range(args.num_seeds):
        seed = random.randint(0, 1000000)
        command_str = 'python MORL_modules/run_algos/run_ppo_psl.py '\
                '--env-name MO_PCSEnergyNet '\
                '--seed {} '\
                '--num-env-steps 30000000 '\
                '--eval-num 1 '\
                '--obj-rms '\
                '--ob-rms '\
                '--raw '.format(seed)
        cmd = command_str+\
        '--warmup-lr 5e-5 '\
        '--psl-lr 5e-5 '\
        '--num-split-obj 5 '\
        '--hypernet-dim 10 '\
        '--reset-logstd '\
        '--W-variance 0 '\
        '--alpha {} '\
        '--save-dir {}/Hyper-MORL/PCS/4-obj{}/{}'\
            .format(0.15, './MORL_modules/results', exp_name, i)
        commands.append(cmd)



def worker(input, output):
    for cmd in iter(input.get, 'STOP'):
        ret_code = os.system(cmd)
        if ret_code != 0:
            output.put('killed')
            break
    output.put('done')
# Create queues
task_queue = Queue()
done_queue = Queue()

# Submit tasks
for cmd in commands:
    task_queue.put(cmd)

# Submit stop signals
for i in range(num_processes):
    task_queue.put('STOP')

# Start worker processes
for i in range(num_processes):
    Process(target=worker, args=(task_queue, done_queue)).start()

# Get and print results
for i in range(num_processes):
    print(f'Process {i}', done_queue.get())