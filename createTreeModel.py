from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import pickle
import os


_data_home = 'data'
_model_home = 'treeModels'
_dataset_ids = [4538, 44, 43174, 1475, 41150, 41145, 41168, 44975, 4549, 1219]
_id2depth = {
    41168 : 40,
    41150 : 20,
    41145 : 15,
    44: 15,
    1475 : 20,
    4538 : 10,
    43174 : 10,
    44975 : 30,
    4549 : 50}
_classification_ids = [41168, 41150, 41145, 44, 1475, 4538, 1219]
_datasets_using_scaler = [44975, 4549]


if 'test' in os.getcwd().split(os.sep):
    _data_home = os.path.join('..', _data_home)
    _model_home = os.path.join('..', _model_home)


def createTreeModel(dataset_id, n_estimators, random_seed, depth=None):
    assert dataset_id in _dataset_ids
    X_train, X_test, y_train, y_test = data_from_openml(dataset_id, random_seed)
    
    if dataset_id in _classification_ids:
        if n_estimators:
            model = GradientBoostingClassifier(n_estimators=n_estimators, 
                                               max_depth=depth or _id2depth[dataset_id], 
                                               random_state=random_seed)
        else:
            model = DecisionTreeClassifier(random_state=random_seed, 
                                           max_depth=depth or _id2depth[dataset_id])
    else:
        if n_estimators:
            model = GradientBoostingRegressor(n_estimators=n_estimators, 
                                              max_depth=depth or _id2depth[dataset_id], 
                                              random_state=random_seed)
        else:
            model = DecisionTreeRegressor(random_state=random_seed,
                                          max_depth=depth or _id2depth[dataset_id])
        
    os.makedirs(_model_home, exist_ok=True)
    if n_estimators:
        model_path = os.path.join(_model_home, 
                                  f'dataset_id={dataset_id}-depth={depth or _id2depth[dataset_id]}-n_estimators={n_estimators}.pkl')
    else:
        model_path = os.path.join(_model_home, 
                                  f'dataset_id={dataset_id}-depth={depth or _id2depth[dataset_id]}.pkl')
        
    if not os.path.exists(model_path):
        model.fit(X_train, y_train)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
                
    return model, X_test, y_test


def data_from_openml(dataset_id, random_seed, test_size=0.2):
    X, y = fetch_openml(data_id=dataset_id, return_X_y=True, as_frame=False, 
                        data_home=_data_home)
    if dataset_id in _classification_ids:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed)
        if dataset_id in _datasets_using_scaler:
            scaler = StandardScaler()
            y_train = scaler.fit_transform(y_train[:, None])[:, 0]
            y_test = scaler.transform(y_test[:, None])[:, 0]
                
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    for dataset_id in _dataset_ids:
        print(dataset_id)
        for n_estimators in [0, 5]:
            createTreeModel(dataset_id, n_estimators, 2025)
            
    
    for depth in range(35, 66):
        createTreeModel(1219, 0, 2025, depth)
        
            