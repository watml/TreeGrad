import numpy as np
from collections import defaultdict
from .treegrad import treegrad
from .treegrad_shap import treegrad_shap
import numbers



def treestab(model, x, semivalue, class_index=None):
    # if class_index is None, model would be treated as regression trees
    if isinstance(semivalue, tuple):
        assert len(semivalue) == 2
        alpha, beta = semivalue
        assert isinstance(alpha, numbers.Integral) and alpha > 0
        assert isinstance(beta, numbers.Integral) and beta > 0
        return treegrad_shap(model, x, semivalue, class_index)
    else:
        assert 0 < semivalue and semivalue < 1
        return treegrad(model, x, np.full(len(x), semivalue, dtype=np.float64), class_index)