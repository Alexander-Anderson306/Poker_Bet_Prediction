from sklearn.ensemble import RandomForestRegressor
import bisect
import numpy as np
from data_prep import separate_players
from joblib import Parallel, delayed

def train_random_forest(X_train, X_test, y_train, y_test, num_trees):
    model = RandomForestRegressor(n_estimators=num_trees, random_state=67).fit(X_train, y_train)
    return model.score(X_test, y_test)

def train_and_score_rf(df, prepare_function, file_prefix, selector=None, extractor=None, k=10):
    num_trees = [100, 1000, 10000]
    dfs = separate_players(df)
    scores = {}
    max_scores = {}
    selected_features_dict = {}
    selected_features = None
    feature_names = None
    feature_name_to_index = None

    if selector is not None and extractor is not None:
        raise Exception("Both feature selection and feature extraction are selected. Both cannot be done")

    for num_t in num_trees:
        scores[num_t] = []
        max_scores[num_t] = 0
    
    def process_player(df):
        scores_local = {}
        max_scores_local = {}
        selected_features_local = {}
        feature_names_local = None

        for num_t in num_trees:
            scores_local[num_t] = []
            max_scores_local[num_t] = 0
            selected_features_local[num_t] = None
        
        for num_t in num_trees:
            selected_features = None
            X_train, X_test, y_train, y_test = None, None, None, None
            if selector is None and extractor is None:
                X_train, X_test, y_train, y_test = prepare_function(df)
            elif selector is not None:
                estimator = RandomForestRegressor(n_estimators=num_t, random_state=67)
                X_train, X_test, y_train, y_test, selected_features, current_feature_names = prepare_function(df, estimator)
                feature_names_local = list(current_feature_names)
            elif extractor is not None:
                X_train, X_test, y_train, y_test = prepare_function(df, k,'linear')
            
            print(f"Training Random Forest with Num Trees={num_t} for player {df['player'].iloc[0]}")

            acc = train_random_forest(X_train, X_test, y_train, y_test, num_t)
            scores_local[num_t].append(acc)
            if acc > max_scores_local[num_t]:
                max_scores_local[num_t] = acc
            selected_features_local[num_t] = selected_features
        
        return scores_local, max_scores_local, selected_features_local, feature_names_local
    
    results = Parallel(n_jobs=-1)(delayed(process_player)(df) for df in dfs)

    for scores_local, max_scores_local, selected_features_local, feature_names_local in results:
        if feature_names is None and feature_names_local is not None:
            feature_names = feature_names_local
            feature_name_to_index = {name: i for i, name in enumerate(feature_names)}
            for existing_key in scores.keys():
                selected_features_dict[existing_key] = [0] * len(feature_names)
        for key in scores_local.keys():
            for acc in scores_local[key]:
                bisect.insort(scores[key], acc)
            if max_scores_local[key] > max_scores[key]:
                max_scores[key] = max_scores_local[key]
            selected_features = selected_features_local[key]
            if selected_features is not None:
                for feature in selected_features:
                    selected_features_dict[key][feature_name_to_index[feature]] += 1
                
    with open(file_prefix+'random_forest_regressor.csv', 'w') as f:
        f.write('num_trees,avg_score,median_score,max_score,selected_features,feature_names\n')
        for num_t in num_trees:
            avg_score = np.mean(scores[num_t])
            median_score = np.median(scores[num_t])
            max_score = max_scores[num_t]
            selected_features = selected_features_dict.get(num_t, None)
            if selected_features is not None:
                selected_features = [count / len(dfs) for count in selected_features]
            f.write(f"{num_t},{avg_score},{median_score},{max_score},{selected_features},{feature_names}\n")