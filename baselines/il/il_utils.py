def flatten_sweep_parameters(param_dict):
    flattened = {}
    for k, v in param_dict.items():
        if isinstance(v, dict):
            if "value" in v:
                flattened[k] = [v["value"]]
            elif "values" in v:
                flattened[k] = v["values"]
        else:
            flattened[k] = [v]
    return flattened