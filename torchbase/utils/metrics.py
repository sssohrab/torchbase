from typing import List, Dict, Any, Callable
from functools import wraps
import inspect
import keyword


def _map_metric(func: Callable, keyword_maps: Dict[str, str]) -> Callable:
    original_sig = inspect.signature(func)
    explicit = {name: param for name, param in original_sig.parameters.items()
                if param.kind == inspect.Parameter.KEYWORD_ONLY}
    extra_params = [param for param in original_sig.parameters.values()
                    if param.kind == inspect.Parameter.VAR_KEYWORD]
    # A group can share a mapping even when its metrics use different arguments.
    mapping = {source: target for source, target in keyword_maps.items() if target in explicit or extra_params}
    if not mapping:
        return func
    invalid_source = next((source for source in mapping if not source.isidentifier() or keyword.iskeyword(source)),
                          None)
    if invalid_source is not None:
        raise ValueError("keyword_maps conflict with metric `{}`: {!r} is not a valid parameter name".format(
            func.__name__, invalid_source))
    reverse_mapping = {target: source for source, target in mapping.items()}
    try:
        parameters = [param.replace(name=reverse_mapping.get(name, name)) for name, param in explicit.items()]
        parameters += [inspect.Parameter(source, inspect.Parameter.KEYWORD_ONLY)
                       for source, target in mapping.items() if target not in explicit]
        mapped_sig = original_sig.replace(parameters=parameters + extra_params)
    except ValueError as error:
        raise ValueError("keyword_maps conflict with metric `{}`: {}".format(func.__name__, error)) from error

    @wraps(func)
    def mapped_function(**kwargs):
        mapped_kwargs = {}
        for name, value in kwargs.items():
            target = mapping.get(name, name)
            if target in mapped_kwargs:
                raise TypeError("Metric `{}` received multiple values for `{}`.".format(func.__name__, target))
            mapped_kwargs[target] = value
        return func(**mapped_kwargs)

    mapped_function.__signature__ = mapped_sig
    return mapped_function


class BaseMetricsClass:
    def __init__(self, keyword_maps: Dict[str, str] | None = None):
        if keyword_maps is not None:
            if not isinstance(keyword_maps, dict) or not all(
                    [isinstance(k, str) and isinstance(v, str) for k, v in keyword_maps.items()]):
                raise TypeError(
                    "The passed `keyword_maps`, if specified, should be a dictionary of string keys and values.")
            self.keyword_maps = dict(keyword_maps)
        else:
            self.keyword_maps = {}

    def get_all_metric_functionals_dict(self) -> Dict[str, Callable[..., Any]]:
        functional_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not attr_name.startswith('get_'):
                attr = getattr(self, attr_name)
                if callable(attr):
                    sig = inspect.signature(attr)
                    if all(param.kind in [inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD]
                           for param in sig.parameters.values()):
                        functional_dict[attr_name] = attr
                    else:
                        raise TypeError("Functional `{}` must have keyword-only arguments.".format(attr_name))

        return functional_dict

    def get_metrics(self, methods: List[str]) -> Dict[str, Callable]:
        """Map input names to metric arguments, preserving unmapped arguments and defaults."""
        if len(set(self.keyword_maps.values())) != len(self.keyword_maps):
            raise ValueError("keyword_maps cannot map multiple keywords to the same metric argument.")
        all_functionals = self.get_all_metric_functionals_dict()
        method_dict = {}

        for method_name in methods:
            if method_name in all_functionals:
                original_function = all_functionals[method_name]

                if self.keyword_maps:
                    method_dict[method_name] = _map_metric(original_function, self.keyword_maps)
                else:
                    method_dict[method_name] = original_function
            else:
                raise ValueError(
                    "Method `{}` is not implemented in the `{}`.".format(method_name, type(self).__name__))

        return method_dict
