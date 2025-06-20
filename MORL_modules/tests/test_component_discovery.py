# MORL_modules/tests/test_component_discovery.py

import sys
import os
import logging
import pytest
import numpy as np
from gymnasium import spaces
import inspect
from typing import Any, Dict, List, Tuple

from MORL_modules.wrappers.mo_pcs_wrapper import MOPCSWrapper
from energy_net.envs.energy_net_v0 import EnergyNetV0
from energy_net.market.pricing.cost_types import CostType
from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern


@pytest.fixture
def real_energynet_env():
    """Create a real EnergyNetV0 environment with minimal configuration"""
    try:
        env = EnergyNetV0(
            controller_name="EnergyNetController",
            controller_module="energy_net.controllers",
            env_config_path='energy_net/configs/environment_config.yaml',
            iso_config_path='energy_net/configs/iso_config.yaml',
            pcs_unit_config_path='energy_net/configs/pcs_unit_config.yaml',
            cost_type=CostType.CONSTANT,
            pricing_policy=PricingPolicy.QUADRATIC,
            demand_pattern=DemandPattern.SINUSOIDAL,
        )
        return env
    except Exception as e:
        pytest.skip(f"Could not create real EnergyNet environment: {e}")


def explore_object_attributes(obj: Any, name: str, max_depth: int = 3, current_depth: int = 0) -> Dict:
    """
    Recursively explore an object's attributes and methods.
    Returns a structured dictionary of findings.
    """
    if current_depth >= max_depth or obj is None:
        return {"type": str(type(obj)), "value": str(obj) if not callable(obj) else "callable"}

    result = {
        "type": str(type(obj)),
        "class_name": obj.__class__.__name__,
        "attributes": {},
        "methods": {},
        "properties": {}
    }

    try:
        # Get all attributes/methods
        attrs = dir(obj)

        for attr_name in attrs:
            # Skip private attributes and common object methods
            if attr_name.startswith('_') and attr_name not in ['__class__', '__dict__']:
                continue

            try:
                attr_value = getattr(obj, attr_name)

                if callable(attr_value):
                    # It's a method
                    try:
                        sig = inspect.signature(attr_value)
                        result["methods"][attr_name] = {
                            "signature": str(sig),
                            "doc": getattr(attr_value, '__doc__', None)
                        }
                    except:
                        result["methods"][attr_name] = {"signature": "unknown"}

                elif isinstance(attr_value, property):
                    # It's a property
                    result["properties"][attr_name] = {
                        "type": str(type(attr_value)),
                        "doc": getattr(attr_value, '__doc__', None)
                    }

                else:
                    # It's a regular attribute
                    if current_depth < max_depth - 1 and hasattr(attr_value, '__dict__'):
                        # Recursively explore complex objects
                        result["attributes"][attr_name] = explore_object_attributes(
                            attr_value, f"{name}.{attr_name}", max_depth, current_depth + 1
                        )
                    else:
                        result["attributes"][attr_name] = {
                            "type": str(type(attr_value)),
                            "value": str(attr_value) if len(str(attr_value)) < 100 else f"{str(attr_value)[:100]}..."
                        }

            except Exception as e:
                result["attributes"][attr_name] = {"error": str(e)}

    except Exception as e:
        result["error"] = str(e)

    return result


def print_component_structure(structure: Dict, indent: int = 0) -> None:
    """Pretty print the component structure."""
    prefix = "  " * indent

    if "class_name" in structure:
        print(f"{prefix}📦 {structure['class_name']} ({structure['type']})")

    # Print important attributes first
    if "attributes" in structure:
        important_attrs = []
        for attr_name, attr_info in structure["attributes"].items():
            if any(keyword in attr_name.lower() for keyword in
                   ['battery', 'pcs', 'energy', 'level', 'state', 'manager', 'unit', 'controller']):
                important_attrs.append((attr_name, attr_info))

        if important_attrs:
            print(f"{prefix}  🔑 Key Attributes:")
            for attr_name, attr_info in important_attrs:
                if "class_name" in attr_info:
                    print(f"{prefix}    ✅ {attr_name}: {attr_info['class_name']}")
                    print_component_structure(attr_info, indent + 3)
                else:
                    print(f"{prefix}    ➡️  {attr_name}: {attr_info.get('type', 'unknown')}")

    # Print useful methods
    if "methods" in structure:
        useful_methods = []
        for method_name, method_info in structure["methods"].items():
            if any(keyword in method_name.lower() for keyword in
                   ['get', 'set', 'update', 'level', 'state', 'battery', 'energy', 'production', 'consumption']):
                useful_methods.append((method_name, method_info))

        if useful_methods:
            print(f"{prefix}  🔧 Key Methods:")
            for method_name, method_info in useful_methods[:10]:  # Limit to first 10
                sig = method_info.get('signature', 'unknown')
                print(f"{prefix}    🎯 {method_name}{sig}")


def check_component_functionality(obj: Any, name: str) -> None:
    """Check specific functionality of components."""

    if name == "Battery" and obj is not None:
        print("  🔋 Battery Tests:")
        try:
            if hasattr(obj, 'energy_level'):
                level = obj.energy_level
                print(f"    ✅ energy_level: {level}")
            if hasattr(obj, 'get_state'):
                state = obj.get_state()
                print(f"    ✅ get_state(): {state}")
            if hasattr(obj, 'energy_max'):
                max_energy = obj.energy_max
                print(f"    ✅ energy_max: {max_energy}")
            if hasattr(obj, 'energy_min'):
                min_energy = obj.energy_min
                print(f"    ✅ energy_min: {min_energy}")
        except Exception as e:
            print(f"    ❌ Battery test error: {e}")

    elif name == "Battery Manager" and obj is not None:
        print("  ⚡ Battery Manager Tests:")
        try:
            if hasattr(obj, 'get_level'):
                level = obj.get_level()
                print(f"    ✅ get_level(): {level}")
            if hasattr(obj, 'get_state'):
                state = obj.get_state()
                print(f"    ✅ get_state(): {state}")
            if hasattr(obj, 'battery_max'):
                max_cap = obj.battery_max
                print(f"    ✅ battery_max: {max_cap}")
        except Exception as e:
            print(f"    ❌ Battery Manager test error: {e}")

    elif name == "PCS Unit" and obj is not None:
        print("  🏭 PCS Unit Tests:")
        try:
            if hasattr(obj, 'get_self_production'):
                production = obj.get_self_production()
                print(f"    ✅ get_self_production(): {production}")
            if hasattr(obj, 'get_self_consumption'):
                consumption = obj.get_self_consumption()
                print(f"    ✅ get_self_consumption(): {consumption}")
            if hasattr(obj, 'battery'):
                battery = obj.battery
                print(f"    ✅ battery: {type(battery).__name__}")
        except Exception as e:
            print(f"    ❌ PCS Unit test error: {e}")

    elif name == "Controller" and obj is not None:
        print("  🎮 Controller Tests:")
        try:
            if hasattr(obj, 'get_battery_level'):
                level = obj.get_battery_level()
                print(f"    ✅ get_battery_level(): {level}")
            if hasattr(obj, 'pcs_unit'):
                pcs = obj.pcs_unit
                print(f"    ✅ pcs_unit: {type(pcs).__name__}")
            if hasattr(obj, 'pcsunit'):
                pcs = obj.pcsunit
                print(f"    ✅ pcsunit: {type(pcs).__name__}")
        except Exception as e:
            print(f"    ❌ Controller test error: {e}")


def generate_recommendations(components_status: Dict) -> None:
    """Generate recommendations for improving the wrapper."""

    available_components = [name for name, obj in components_status.items() if obj is not None]
    missing_components = [name for name, obj in components_status.items() if obj is None]

    print("📝 Based on the analysis:")
    print()

    if available_components:
        print("✅ AVAILABLE COMPONENTS:")
        for comp in available_components:
            print(f"   • {comp}")
        print()

    if missing_components:
        print("❌ MISSING COMPONENTS:")
        for comp in missing_components:
            print(f"   • {comp}")
        print()

    print("🔧 SUGGESTED IMPROVEMENTS:")

    # Analyze the pattern and suggest standardization
    if components_status["Controller"] is not None:
        print("   1. ✅ Controller is available - can be used as main entry point")

        if components_status["Battery Manager"] is not None:
            print("   2. ✅ Use Battery Manager for all battery operations")
            print("      - Eliminates need for hasattr checks on battery")

        if components_status["PCS Unit"] is not None:
            print("   3. ✅ Use PCS Unit for production/consumption")
            print("      - Standardize on get_self_production/get_self_consumption")

    print("   4. 🎯 Recommended wrapper refactoring:")
    print("      - Cache component references after first successful lookup")
    print("      - Use try/except instead of hasattr for better performance")
    print("      - Create component interface adapter")
    print("      - Add component validation at wrapper initialization")


def test_comprehensive_component_discovery(real_energynet_env):
    """
    Comprehensive test that discovers and reports all available components,
    their attributes, and methods in the EnergyNet environment.
    """
    print("\n" + "=" * 80)
    print("🔍 COMPREHENSIVE COMPONENT DISCOVERY REPORT")
    print("=" * 80)

    wrapper = MOPCSWrapper(
        real_energynet_env,
        num_objectives=4,
        log_level=logging.WARNING  # Reduce noise
    )

    # Reset environment to ensure it's properly initialized
    obs, info = wrapper.reset()
    print("✅ Environment reset successfully")

    # Extract components using the wrapper's method
    wrapper._get_environment_components()

    print("\n📊 COMPONENT AVAILABILITY SUMMARY:")
    print("-" * 50)

    components_status = {
        "Environment (env)": wrapper.env,
        "Unwrapped Environment": getattr(wrapper.env, 'unwrapped', None),
        "Controller": wrapper.controller,
        "PCS Unit": wrapper.pcsunit,
        "Battery": wrapper.battery,
        "Battery Manager": wrapper.battery_manager,
    }

    # Print availability summary
    for comp_name, comp_obj in components_status.items():
        status = "✅ Available" if comp_obj is not None else "❌ Not Found"
        comp_type = type(comp_obj).__name__ if comp_obj is not None else "None"
        print(f"{comp_name}: {status} ({comp_type})")

    print("\n" + "=" * 80)
    print("📖 DETAILED COMPONENT ANALYSIS")
    print("=" * 80)

    # Detailed exploration of each component
    for comp_name, comp_obj in components_status.items():
        if comp_obj is not None:
            print(f"\n🔬 ANALYZING: {comp_name}")
            print("-" * 60)

            structure = explore_object_attributes(comp_obj, comp_name, max_depth=3)
            print_component_structure(structure)

            # Test specific functionality
            print(f"\n🧪 FUNCTIONALITY TESTS for {comp_name}:")
            check_component_functionality(comp_obj, comp_name)

    print("\n" + "=" * 80)
    print("🎯 RECOMMENDATIONS FOR WRAPPER IMPROVEMENT")
    print("=" * 80)

    # Generate recommendations based on findings
    generate_recommendations(components_status)

    # The test should pass if we get here
    assert True, "Component discovery completed successfully"


if __name__ == "__main__":
    # Run the test directly
    import sys

    sys.path.append('.')

    # Create environment and run discovery
    try:
        env = EnergyNetV0(
            controller_name="EnergyNetController",
            controller_module="energy_net.controllers",
            env_config_path='energy_net/configs/environment_config.yaml',
            iso_config_path='energy_net/configs/iso_config.yaml',
            pcs_unit_config_path='energy_net/configs/pcs_unit_config.yaml',
            cost_type=CostType.CONSTANT,
            pricing_policy=PricingPolicy.QUADRATIC,
            demand_pattern=DemandPattern.SINUSOIDAL,
        )

        wrapper = MOPCSWrapper(env, num_objectives=4, log_level=logging.WARNING)
        test_comprehensive_component_discovery(env)

    except Exception as e:
        print(f"❌ Could not run discovery test: {e}")
        import traceback

        traceback.print_exc()