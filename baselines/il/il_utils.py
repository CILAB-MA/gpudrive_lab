def flatten_parameters(parameters):
    flattened = {}
    for key, val in parameters.items():
        if isinstance(val, dict):
            if 'value' in val:
                flattened[key] = val['value']
            elif 'values' in val:
                flattened[key] = val['values'][0] if val['values'] else None
            else:
                raise ValueError(f"Unknown format for key '{key}': {val}")
        else:
            raise ValueError(f"Expected dict for key '{key}', got {type(val)}")
    return flattened