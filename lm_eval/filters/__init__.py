from lm_eval.api.filter import FilterEnsemble
from . import selection
from . import extraction


FILTER_REGISTRY = {
    "take_first": selection.TakeFirstFilter,
    "regex": extraction.RegexFilter,
    # TODO: implement this filter. either it should take in an arbitrary "scoring"/reward function
    # that takes an input and returns a scalar and then should select the max reward,
    # or should implement different filters for different ways of handling a reward model's inference.
    # "arg_max": selection.ArgMaxFilter,
}


def get_filter(filter_name):
    return FILTER_REGISTRY[filter_name]


def build_filter_ensemble(filter_name, components):
    """
    Create a filtering pipeline.
    """

    filters = []
    for (function, kwargs) in components:
        if kwargs is None:
            f = get_filter(function)()
        else:
            # create a filter given its name in the registry
            f = get_filter(function)(**kwargs)  # TODO: pass kwargs to filters properly
        # add the filter as a pipeline step
        filters.append(f)

    return FilterEnsemble(name=filter_name, filters=filters)