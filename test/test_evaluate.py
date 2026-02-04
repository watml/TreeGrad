import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utilFuncs import treeUtility
from createTreeModel import createTreeModel
import numpy as np

# test binary classification
model, X_test, y_test = createTreeModel(44, 0, None)
print(model.predict_proba(X_test[0, None]))
for i in range(2):
    util = treeUtility(model, X_test[0], i)
    print(util.evaluate(np.ones(X_test.shape[1], dtype=bool)))


model, X_test, y_test = createTreeModel(44, 2, None)
print(model.predict_proba(X_test[0, None]))
print(model.decision_function(X_test[0, None]))
for i in range(2):
    util = treeUtility(model, X_test[0], i)
    print(util.evaluate(np.ones(X_test.shape[1], dtype=bool), test=True))
    print(util.evaluate(np.ones(X_test.shape[1], dtype=bool)))



# test multiclass classification
model, X_test, y_test = createTreeModel(1475, 0, None)
print(model.predict_proba(X_test[-3, None]))
for i in range(6):
    util = treeUtility(model, X_test[-3], i)
    print(util.evaluate(np.ones(X_test.shape[1], dtype=bool)))


model, X_test, y_test = createTreeModel(1475, 2, None)
print(model.decision_function(X_test[-3, None]))
print(model.predict_proba(X_test[-3, None]))
for i in range(6):
    util = treeUtility(model, X_test[-3], i)
    print(util.evaluate(np.ones(X_test.shape[1], dtype=bool)))
    print(util.evaluate(np.ones(X_test.shape[1], dtype=bool), test=True))


# test regression
model, X_test, y_test = createTreeModel(43174, 0, None)
util = treeUtility(model, X_test[0])
print(model.predict(X_test[0, None]))
print(util.evaluate(np.ones(X_test.shape[1], dtype=bool)))


model, X_test, y_test = createTreeModel(43174, 2, None)
util = treeUtility(model, X_test[0])
print(model.predict(X_test[0, None]))
print(util.evaluate(np.ones(X_test.shape[1], dtype=bool)))
print(util.evaluate(np.ones(X_test.shape[1], dtype=bool), test=True))

