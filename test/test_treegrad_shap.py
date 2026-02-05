import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TreeGrad import treegrad_shap
import numpy as np



def test_treegrad_shap(util, model, x, class_index=None):
    t1 = treegrad_shap(model, x, class_index)
    # (1,1) corresponds to the Shapley value
    t2 = util.groundtruth_bruteforce((1,1))
    print(np.linalg.norm(t1-t2))




if __name__ == '__main__':
    from sklearn.datasets import make_regression, make_classification
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from utilFuncs import treeUtility
    
    # test regression
    n_features = 10
    x, y = make_regression(1000, n_features=n_features)
    model = GradientBoostingRegressor(n_estimators=5, max_depth=5).fit(x, y)
    util = treeUtility(model, x[0])
    test_treegrad_shap(util, model, x[0])
    
    
    model = DecisionTreeRegressor(max_depth=5).fit(x, y)
    util = treeUtility(model, x[0])
    test_treegrad_shap(util, model, x[0])
    
    
    # test binary classification
    x, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        random_state=0
    )
    model = GradientBoostingClassifier(n_estimators=5, max_depth=5).fit(x, y)
    util = treeUtility(model, x[0], 0)
    test_treegrad_shap(util, model, x[0], 0)
    
    model = DecisionTreeClassifier(max_depth=5).fit(x, y)
    util = treeUtility(model, x[0], 1)
    test_treegrad_shap(util, model, x[0], 1)
    
    
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
    test_treegrad_shap(util, model, x[0], 3)
    
    model = DecisionTreeClassifier(max_depth=5).fit(x, y)
    util = treeUtility(model, x[0], 3)
    test_treegrad_shap(util, model, x[0], 3)