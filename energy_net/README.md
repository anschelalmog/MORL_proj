# EnergyNet: Multi-Agent Reinforcement Learning for Smart Grid Simulation

EnergyNet is a framework for simulating smart grid environments and training reinforcement learning agents to optimize grid operations. The framework features a multi-agent environment with two key strategic entities: the Independent System Operator (ISO) and Production Consumption Storage (PCS) agents.

## System Overview

### Key Components

1. **Independent System Operator (ISO)**
   - Sets energy prices (buy/sell)
   - Controls dispatch levels
   - Pricing Policies: ONLINE, QUADRATIC, or CONSTANT

2. **Power Control System (PCS)**
   - Controls battery storage systems
   - Decides when to charge/discharge
   - Responds to price signals

3. **Environment (EnergyNetV0)**
   - Multi-agent environment
   - Handles sequential interactions
   - Manages shared state
   - Calculates rewards

## Installation

```bash
# Clone the repository
git clone https://github.com/CLAIR-LAB-TECHNION/energy-net
cd EnergyNet

# Install package
pip install -e .
```

## Testing

To run the test suite, use the following command:

```bash
python -m unittest discover tests
```

This will run all test files in the `tests` directory. For more detailed test output, you can use:

```bash
python -m unittest discover tests -v
```

To run a specific test file, you can use:

```bash
python -m unittest tests/test_environment_registration.py
```

## Configuration

### Main Configuration Files

1. **environment_config.yaml**
   - Time parameters
   - Pricing parameters
   - Demand prediction

2. **iso_config.yaml**
   - Pricing ranges
   - Dispatch settings
   - Action space parameters

3. **pcs_unit_config.yaml**
   - Battery parameters
   - Action space parameters
   - Energy unit settings

### Environment Parameters

```python
env = EnergyNetV0(
    controller_name="EnergyNetController",  # Name of the controller class
    controller_module="energy_net.controllers",  # Module containing the controller
    cost_type="CONSTANT",         # CONSTANT, VARIABLE, TIME_OF_USE
    pricing_policy="ONLINE",      # ONLINE, QUADRATIC, CONSTANT
    demand_pattern="SINUSOIDAL",  # SINUSOIDAL, RANDOM, PERIODIC, SPIKES
    use_dispatch_action=True      # Enable dispatch control
)
```

## Reward Structure

1. **ISO Rewards**
   - Minimize reserve costs
   - Minimize dispatch costs
   - Avoid demand shortfalls

2. **PCS Rewards**
   - Profit from energy arbitrage
   - Buy low, sell high
   - Efficient battery usage




## Running tests:
pip install -e .[test] 
