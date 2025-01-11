from typing import Tuple, List, Dict, Any, Callable
import inspect


class BaseMetricsClass:
    def __init__(self, keyword_maps: Dict[str, str] | None = None):
        if keyword_maps is not None:
            if not isinstance(keyword_maps, dict) or not all(
                    [isinstance(k, str) and isinstance(v, str) for k, v in keyword_maps.items()]):
                raise TypeError(
                    "The passed `keyword_maps`, if specified, should be a dictionary of string keys and values.")
            self.keyword_maps = keyword_maps
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
        all_functionals = self.get_all_metric_functionals_dict()
        method_dict = {}

        for method_name in methods:
            if method_name in all_functionals:
                original_function = all_functionals[method_name]

                if self.keyword_maps:
                    def create_mapped_function(func, keyword_maps):
                        def mapped_function(**kwargs):
                            mapped_kwargs = {keyword_maps.get(k, k): v for k, v in kwargs.items()}
                            return func(**mapped_kwargs)

                        # Create new signature with mapped keyword arguments
                        original_sig = inspect.signature(func)
                        new_params = [inspect.Parameter(new_key, inspect.Parameter.KEYWORD_ONLY) for new_key in
                                      keyword_maps]
                        new_sig = original_sig.replace(parameters=new_params)
                        mapped_function.__signature__ = new_sig

                        return mapped_function

                    method_dict[method_name] = create_mapped_function(original_function, self.keyword_maps)
                else:
                    method_dict[method_name] = original_function
            else:
                raise ValueError(
                    "Method `{}` is not implemented in the `{}`.".format(method_name, type(self).__name__))

        return method_dict
