import yaml
from types import SimpleNamespace
from typing import Any, Dict

class _SafeDict(dict):
    """
    A dictionary subclass that returns '{key}' for missing keys when used
    with str.format_map(). This prevents KeyError for placeholders.
    """
    def __missing__(self, key: str) -> str:
        return f'{{{key}}}'

def _recursive_format(item: Any, context: dict) -> Any:
    """
    Recursively traverses a nested structure (dicts, lists), formats string
    values using the context, and converts any string 'None' to the
    Python None object.
    """
    if isinstance(item, dict):
        return {k: _recursive_format(v, context) for k, v in item.items()}
    elif isinstance(item, list):
        return [_recursive_format(v, context) for v in item]
    elif isinstance(item, str):
        if item == 'None':
            return None
        return item.format_map(context)
    else:
        return item

def _to_hybrid_namespace(d: Dict) -> SimpleNamespace:
    """
    Converts only the top level of a dictionary to a SimpleNamespace.
    Nested dictionaries and lists remain as they are.
    """
    ns = SimpleNamespace()
    for key, value in d.items():
        setattr(ns, key, value)
    return ns

def parse_config(yaml_path: str) -> SimpleNamespace:
    """
    Parses a YAML configuration file into a hybrid namespace where only the
    top-level keys are accessible via dot notation. It resolves self-referential
    string templates and converts string values of 'None' to NoneType.

    Args:
        yaml_path: The path to the input YAML file.

    Returns:
        A SimpleNamespace object with nested dictionaries.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # 1. Separate base config from the models list
    base_config = {k: v for k, v in data.items() if k != 'model_params'}
    model_params_list = data.get('model_params', [])

    # 2. Iteratively format the base configuration
    for _ in range(5):
        context = _SafeDict(base_config)
        base_config = _recursive_format(base_config, context)

    # 3. Process each dictionary within the model_params list
    processed_models = []
    for model_dict in model_params_list:
        full_context = _SafeDict({**base_config, **model_dict})
        formatted_model = _recursive_format(model_dict, full_context)
        processed_models.append(formatted_model)

    # 4. Recombine into the final configuration dictionary
    final_config_dict = base_config
    final_config_dict['model_params'] = processed_models

    # 5. Convert only the top level to a SimpleNamespace
    return _to_hybrid_namespace(final_config_dict)