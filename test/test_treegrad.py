import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import itertools


def compute_gradient(game, z):
    n_players = game.n_players
    assert n_players == len(z)   
    zc = 1 - z
    gradient = np.zeros(n_players, dtype=np.float64)
    for subset in itertools.product([True, False], repeat=n_players):
        subset = np.array(subset)
        result = game.evaluate(subset)      
        for player in range(n_players):
            if subset[player]:
                weight = zc[~subset].prod()
                subset[player] = 0
                weight *= z[subset].prod()
                subset[player] = 1
                gradient[player] += weight * result 
            else:
                weight = z[subset].prod()
                subset[player] = 1
                weight *= zc[~subset].prod()
                subset[player] = 0
                gradient[player] -= weight * result              
    return gradient


def test_treegrad(util, model, x, class_index=None):
    t1 = compute_gradient(util, np.full(10, 0.5))
    t2 = treegrad(model, x, np.full(10, 0.5), class_index)
    print(np.linalg.norm(t1-t2))
    t1 = compute_gradient(util, np.full(10, 1))
    t2 = treegrad(model, x, np.full(10, 1), class_index)
    print(np.linalg.norm(t1-t2))
    err = np.empty(10)
    for i in range(10):
        z = np.random.choice([1, 0], size=10)
        t1 = compute_gradient(util, z)
        t2 = treegrad(model, x, z, class_index)
        err[i] = np.linalg.norm(t1-t2)
    print(err.mean())



if __name__ == '__main__':
    from sklearn.datasets import make_regression, make_classification
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from TreeGrad import treegrad
    from utilFuncs import treeUtility
    import numpy as np
    
    # test regression
    n_features = 10
    x, y = make_regression(1000, n_features=n_features)
    model = GradientBoostingRegressor(n_estimators=5, max_depth=5).fit(x, y)
    util = treeUtility(model, x[0])
    test_treegrad(util, model, x[0])
    
    
    model = DecisionTreeRegressor(max_depth=5).fit(x, y)
    util = treeUtility(model, x[0])
    test_treegrad(util, model, x[0])
    
    
    # test binary classification
    x, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        random_state=0
    )
    model = GradientBoostingClassifier(n_estimators=5, max_depth=5).fit(x, y)
    util = treeUtility(model, x[0], 0)
    test_treegrad(util, model, x[0], 0)
    
    model = DecisionTreeClassifier(max_depth=5).fit(x, y)
    util = treeUtility(model, x[0], 1)
    test_treegrad(util, model, x[0], 1)
    
    
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
    test_treegrad(util, model, x[0], 3)
    
    model = DecisionTreeClassifier(max_depth=5).fit(x, y)
    util = treeUtility(model, x[0], 2)
    test_treegrad(util, model, x[0], 2)