# TODO:
# run a default PCS agent -> and get the info from the energy net
# https://github.com/CLAIR-LAB-TECHNION/energy-net/tree/main/energy_net/model/rewards
# decide and implement how to compute each reward
# this file could help us VVVVVV
# https://github.com/CLAIR-LAB-TECHNION/energy-net/blob/main/energy_net/controllers/pcs/battery_manager.py
#
# Critic 1 -> max profit
# Critic 2 -> bound actions on battery (try to minimize)
# (Optional): Critic 3 -> max utilization of own productive energy <-> min buying operations
#


# action = battery(t)-battery(t-1)
# act