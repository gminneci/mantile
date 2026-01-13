import json
import os
from pathlib import Path
from typing import Dict, List
from .models import HardwareSpecs

# Get the hardware_configs directory path
HARDWARE_CONFIGS_DIR = Path(__file__).parent / "data" / "hardware_configs"


def load_hardware_config(config_name: str) -> HardwareSpecs:
    """
    Load hardware configuration from JSON file.
    
    Args:
        config_name: Name of the config file (with or without .json extension)
                    Examples: "nvidia_gb200_single", "nvidia_nvl72_rack"
    
    Returns:
        HardwareSpecs object loaded from JSON
    
    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the JSON is invalid or missing required fields
    """
    # Add .json extension if not present
    if not config_name.endswith('.json'):
        config_name = f"{config_name}.json"
    
    config_path = HARDWARE_CONFIGS_DIR / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Hardware config '{config_name}' not found in {HARDWARE_CONFIGS_DIR}. "
            f"Available configs: {', '.join(list_available_configs())}"
        )
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Validate and create HardwareSpecs
    try:
        return HardwareSpecs(**config_data)
    except Exception as e:
        raise ValueError(f"Invalid hardware config '{config_name}': {e}")


def list_available_configs() -> List[str]:
    """
    List all available hardware configuration files.
    
    Returns:
        List of config names (without .json extension)
    """
    if not HARDWARE_CONFIGS_DIR.exists():
        return []
    
    configs = []
    for file_path in HARDWARE_CONFIGS_DIR.glob("*.json"):
        configs.append(file_path.stem)
    
    return sorted(configs)


def get_all_configs() -> Dict[str, HardwareSpecs]:
    """
    Load all available hardware configurations.
    
    Returns:
        Dictionary mapping config names to HardwareSpecs objects
    """
    configs = {}
    for config_name in list_available_configs():
        try:
            configs[config_name] = load_hardware_config(config_name)
        except Exception as e:
            print(f"Warning: Failed to load config '{config_name}': {e}")
    
    return configs


# Backward compatibility functions
def get_nvl72_specs() -> HardwareSpecs:
    """
    Returns the hardware specifications for a SINGLE GB200 package.
    
    DEPRECATED: Use load_hardware_config("nvidia_gb200_single") instead.
    """
    return load_hardware_config("nvidia_gb200_single")


def get_nvl72_rack_specs() -> HardwareSpecs:
    """
    Returns the specs for the FULL NVL-72 Rack.
    
    DEPRECATED: Use load_hardware_config("nvidia_nvl72_rack") instead.
    """
    return load_hardware_config("nvidia_nvl72_rack")
