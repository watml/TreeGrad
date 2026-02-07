import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from utilFuncs import treeUtility



#from TreeProb import treeprob
#from TreeProb import treeprob_worsetime as treeprob
from TreeGrad import treestab as treeprob


def test_treeprob(model, util, x, class_index=None):
    for semivalue in [(1,16), (16, 1), (8, 4), (4, 8), (1, 1), 0.2, 0.5, 0.8]:
        t1 = treeprob(model, x, semivalue, class_index)
        t2 = util.groundtruth_bruteforce(semivalue)
        print(np.linalg.norm(t1-t2))


if __name__ == '__main__':
    # test regression
    n_features = 10
    x, y = make_regression(1000, n_features=n_features)
    model = GradientBoostingRegressor(n_estimators=5, max_depth=5).fit(x, y)
    util = treeUtility(model, x[0])
    test_treeprob(model, util, x[0])
    
    model = DecisionTreeRegressor(max_depth=5).fit(x, y)
    util = treeUtility(model, x[0])
    test_treeprob(model, util, x[0])
    
    
    # test binary classification
    x, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        random_state=0
    )
    model = GradientBoostingClassifier(n_estimators=5, max_depth=5).fit(x, y)
    util = treeUtility(model, x[0], 0)
    test_treeprob(model, util, x[0], 0)
    
    
    model = DecisionTreeClassifier(max_depth=5).fit(x, y)
    util = treeUtility(model, x[0], 1)
    test_treeprob(model, util, x[0], 1)
    
    
    
    # test multiclass classification
    x, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=0
    )
    model = GradientBoostingClassifier(n_estimators=5, max_depth=5).fit(x, y)
    util = treeUtility(model, x[0], 3)
    test_treeprob(model, util, x[0], 3)
    
    
    model = DecisionTreeClassifier(max_depth=5).fit(x, y)
    util = treeUtility(model, x[0], 2)
    test_treeprob(model, util, x[0], 2)







