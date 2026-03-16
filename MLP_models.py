from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import bisect
import numpy as np
from data_prep import separate_players, separate_clusters
from joblib import Parallel, delayed

def train_MLP(X_train, X_test, y_train, y_test, learning_rate_init, learning_rate_type, solver):
    model = MLPRegressor(learning_rate_init=learning_rate_init, learning_rate=learning_rate_type, solver=solver, random_state=67, max_iter=3000).fit(X_train, y_train)
    return model.score(X_test, y_test)

def train_and_score_MLP(df, prepare_function, file_prefix, cluster=False, selector=None, extractor=None, k=10):
    dfs = None
    if not cluster:
        dfs = separate_players(df)
    else:
        dfs = separate_clusters(df)
    learning_rates = ['constant', 'invscaling', 'adaptive']
    learning_rate_inits = [0.0001, 0.001, 0.01]
    solvers = ['adam', 'lbfgs']
    MLP_scores = {}
    max_MLP = {}
    selected_features_dict = {}
    selected_features = None
    feature_names = None
    feature_name_to_index = None

    if selector is not None and extractor is not None:
        raise Exception("Both feature selection and feature extraction are selected. Both cannot be done")

    for learning_rate_type in learning_rates:
        for learning_rate_init in learning_rate_inits:
            for solver in solvers:
                key = "{},{},{}".format(learning_rate_type, learning_rate_init, solver)
                MLP_scores[key] = []
                max_MLP[key] = 0

    def process_player(df):
        MLP_local = {}
        selected_features_local = {}
        feature_names_local = None

        for learning_rate_type in learning_rates:
            for learning_rate_init in learning_rate_inits:
                for solver in solvers:
                    key = "{},{},{}".format(learning_rate_type, learning_rate_init, solver)
                    MLP_local[key] = []
                    selected_features_local[key] = None

        X_train, X_test, y_train, y_test = None, None, None, None
        if(selector is None and extractor is None):
            X_train, X_test, y_train, y_test = prepare_function(df)
        for learning_rate_type in learning_rates:
            for learning_rate_init in learning_rate_inits:
                for solver in solvers:
                    selected_features = None
                    key = "{},{},{}".format(learning_rate_type, learning_rate_init, solver)
                    if(selector is not None):
                        estimator = RandomForestRegressor(n_estimators=100, random_state=67)
                        X_train, X_test, y_train, y_test, selected_features, current_feature_names = prepare_function(df, estimator)
                        feature_names_local = list(current_feature_names)
                    elif(extractor is not None):
                        X_train, X_test, y_train, y_test = prepare_function(df, k, 'linear')
                    
                    acc = train_MLP(X_train, X_test, y_train, y_test, learning_rate_init, learning_rate_type, solver)
                    MLP_local[key].append(acc)
                    selected_features_local[key] = selected_features

        return MLP_local, selected_features_local, feature_names_local

    results = Parallel(n_jobs=-1)(delayed(process_player)(df) for df in dfs)

    for MLP_local, selected_features_local, feature_names_local in results:
        if feature_names is None and feature_names_local is not None:
            feature_names = feature_names_local
            feature_name_to_index = {name: i for i, name in enumerate(feature_names)}
            for existing_key in MLP_scores.keys():
                selected_features_dict[existing_key] = [0] * len(feature_names)
        for key in MLP_local.keys():
            for acc in MLP_local[key]:
                bisect.insort(MLP_scores[key], acc)
                if acc > max_MLP[key]:
                    max_MLP[key] = acc
            selected_features = selected_features_local[key]
            if selected_features is not None:
                for feature in selected_features:
                    selected_features_dict[key][feature_name_to_index[feature]] += 1

    with open(file_prefix+'MLP_scores.csv', 'w') as f:
        if not cluster:
            f.write('learning_rate,learning_rate_init,solver,avg_score,median_score,max_score,selected_features,feature_names\n')
            for key in MLP_scores.keys():
                learning_rate_type, learning_rate_init, solver = key.split(',')
                avg_score = np.mean(MLP_scores[key])
                median_score = np.median(MLP_scores[key])
                max_score = max_MLP[key]
                selected_features = selected_features_dict.get(key, None)
                if selected_features is not None:
                    selected_features = [count / len(dfs) for count in selected_features]
                f.write(f"{learning_rate_type},{learning_rate_init},{solver},{avg_score},{median_score},{max_score},{selected_features},{feature_names}\n")
        else:
            f.write('learning_rate,learning_rate_init,solver,c1_score,c2_score,selected_features,feature_names\n')
            for key in MLP_scores.keys():
                learning_rate_type, learning_rate_init, solver = key.split(',')
                selected_features = selected_features_dict.get(key, None)
                if selected_features is not None:
                    selected_features = [count / len(dfs) for count in selected_features]
                f.write(f"{learning_rate_type},{learning_rate_init},{solver},{MLP_scores[key][0]},{MLP_scores[key][1]},{selected_features},{feature_names}\n")