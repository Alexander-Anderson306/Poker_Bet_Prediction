from sklearn.svm import SVR
import bisect
import numpy as np
from data_prep import separate_players
from joblib import Parallel, delayed

################################################################## BELOW FUNCTIONS ARE USED TO TRAIN MODELS ##################################################################
def train_linear_kernel_svr(X_train, X_test, y_train, y_test, C, epsilon):
    model = SVR(kernel='linear', C=C, epsilon=epsilon).fit(X_train, y_train)
    return model.score(X_test, y_test)

def train_poly_kernel_svr(X_train, X_test, y_train, y_test, degree, C, epsilon):
    model = SVR(kernel='poly', degree=degree, C=C, epsilon=epsilon).fit(X_train, y_train)
    return model.score(X_test, y_test)

def train_rbf_kernel_svr(X_train, X_test, y_train, y_test, gamma, C, epsilon):
    model = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon).fit(X_train, y_train)
    return model.score(X_test, y_test)

def train_sigmoid_kernel_svr(X_train, X_test, y_train, y_test, C, epsilon):
    model = SVR(kernel='sigmoid', C=C, epsilon=epsilon).fit(X_train, y_train)
    return model.score(X_test, y_test)

################################################################## BELOW FUNCTIONS PERFORM HYPER PARAMETER OPTOMIZATION ##################################################################
def train_and_score_models(df, prepare_function, file_prefix):
    dfs = separate_players(df)
    C_Vals = [0.1, 1, 10]
    epsilon_Vals = [0.01, 0.1, 1]
    degree_Vals = [2, 3, 4]
    gamma_Vals = ['scale', 'auto']
    linear_scores = {}
    poly_scores = {}
    rbf_scores = {}
    sigmoid_scores = {}
    max_linear = {}
    max_poly = {}
    max_rbf = {}
    max_sigmoid = {}

    for C in C_Vals:
        for epsilon in epsilon_Vals:
            key = "{},{}".format(C, epsilon)
            linear_scores[key] = []
            max_linear[key] = 0
            sigmoid_scores[key] = []
            max_sigmoid[key] = 0
            for degree in degree_Vals:
                poly_key = "{},{},{}".format(degree, C, epsilon)
                poly_scores[poly_key] = []
                max_poly[poly_key] = 0
            for gamma in gamma_Vals:
                rbf_key = "{},{},{}".format(gamma, C, epsilon)
                rbf_scores[rbf_key] = []
                max_rbf[rbf_key] = 0

    def process_player(df):
        linear_local = {}
        poly_local = {}
        rbf_local = {}
        sigmoid_local = {}

        for C in C_Vals:
            for epsilon in epsilon_Vals:
                key = "{},{}".format(C, epsilon)
                linear_local[key] = []
                sigmoid_local[key] = []
                for degree in degree_Vals:
                    poly_key = "{},{},{}".format(degree, C, epsilon)
                    poly_local[poly_key] = []
                for gamma in gamma_Vals:
                    rbf_key = "{},{},{}".format(gamma, C, epsilon)
                    rbf_local[rbf_key] = []

        X_train, X_test, y_train, y_test = prepare_function(df)
        for C in C_Vals:
            for epsilon in epsilon_Vals:
                key = "{},{}".format(C, epsilon)
                print(f"Training linear kernel SVR with C={C} and epsilon={epsilon} for player {df['player'].iloc[0]}")
                acc = train_linear_kernel_svr(X_train, X_test, y_train, y_test, C, epsilon)
                linear_local[key].append(acc)

                for degree in degree_Vals:
                    print(f"Training polynomial kernel SVR with degree={degree}, C={C} and epsilon={epsilon} for player {df['player'].iloc[0]}")
                    poly_key = "{},{},{}".format(degree, C, epsilon)
                    acc = train_poly_kernel_svr(X_train, X_test, y_train, y_test, degree, C, epsilon)
                    poly_local[poly_key].append(acc)

                for gamma in gamma_Vals:
                    print(f"Training RBF kernel SVR with gamma={gamma}, C={C} and epsilon={epsilon} for player {df['player'].iloc[0]}")
                    rbf_key = "{},{},{}".format(gamma, C, epsilon)
                    acc = train_rbf_kernel_svr(X_train, X_test, y_train, y_test, gamma, C, epsilon)
                    rbf_local[rbf_key].append(acc)

                print(f"Training sigmoid kernel SVR with C={C} and epsilon={epsilon} for player {df['player'].iloc[0]}")
                acc = train_sigmoid_kernel_svr(X_train, X_test, y_train, y_test, C, epsilon)
                sigmoid_local[key].append(acc)

        return linear_local, poly_local, rbf_local, sigmoid_local

    results = Parallel(n_jobs=-1)(delayed(process_player)(df) for df in dfs)

    for linear_local, poly_local, rbf_local, sigmoid_local in results:
        for key in linear_local.keys():
            for acc in linear_local[key]:
                bisect.insort(linear_scores[key], acc)
                if acc > max_linear[key]:
                    max_linear[key] = acc
        for key in poly_local.keys():
            for acc in poly_local[key]:
                bisect.insort(poly_scores[key], acc)
                if acc > max_poly[key]:
                    max_poly[key] = acc
        for key in rbf_local.keys():
            for acc in rbf_local[key]:
                bisect.insort(rbf_scores[key], acc)
                if acc > max_rbf[key]:
                    max_rbf[key] = acc
        for key in sigmoid_local.keys():
            for acc in sigmoid_local[key]:
                bisect.insort(sigmoid_scores[key], acc)
                if acc > max_sigmoid[key]:
                    max_sigmoid[key] = acc
    
    with open(file_prefix+'svr_linear_scores.csv', 'w') as f:
        f.write('C,epsilon,avg_score,median_score,max_score\n')
        for key in linear_scores.keys():
            C, epsilon = key.split(',')
            avg_score = np.mean(linear_scores[key])
            median_score = np.median(linear_scores[key])
            max_score = max_linear[key]
            f.write(f"{C},{epsilon},{avg_score},{median_score},{max_score}\n")
    
    with open(file_prefix+'svr_poly_scores.csv', 'w') as f:
        f.write('degree,C,epsilon,avg_score,median_score,max_score\n')
        for key in poly_scores.keys():
            degree, C, epsilon = key.split(',')
            avg_score = np.mean(poly_scores[key])
            median_score = np.median(poly_scores[key])
            max_score = max_poly[key]
            f.write(f"{degree},{C},{epsilon},{avg_score},{median_score},{max_score}\n")
    
    with open(file_prefix+'svr_rbf_scores.csv', 'w') as f:
        f.write('gamma,C,epsilon,avg_score,median_score,max_score\n')
        for key in rbf_scores.keys():
            gamma, C, epsilon = key.split(',')
            avg_score = np.mean(rbf_scores[key])
            median_score = np.median(rbf_scores[key])
            max_score = max_rbf[key]
            f.write(f"{gamma},{C},{epsilon},{avg_score},{median_score},{max_score}\n")
    
    with open(file_prefix+'svr_sigmoid_scores.csv', 'w') as f:
        f.write('C,epsilon,avg_score,median_score,max_score\n')
        for key in sigmoid_scores.keys():
            C, epsilon = key.split(',')
            avg_score = np.mean(sigmoid_scores[key])
            median_score = np.median(sigmoid_scores[key])
            max_score = max_sigmoid[key]
            f.write(f"{C},{epsilon},{avg_score},{median_score},{max_score}\n")

def train_and_score_linear_kernel_svr(df, prepare_function, file_prefix, selector=None, extractor=None, k=10):
    dfs = separate_players(df)
    C_Vals = [0.1, 1, 10]
    epsilon_Vals = [0.01, 0.1, 1]
    linear_scores = {}
    max_linear = {}
    selected_features_dict = {}
    selected_features = None
    feature_names = None
    feature_name_to_index = None

    if selector is not None and extractor is not None:
        raise Exception("Both feature selection and feature extraction are selected. Both cannot be done")

    for C in C_Vals:
        for epsilon in epsilon_Vals:
            key = "{},{}".format(C, epsilon)
            linear_scores[key] = []
            max_linear[key] = 0

    def process_player(df):
        linear_local = {}
        selected_features_local = {}
        feature_names_local = None

        for C in C_Vals:
            for epsilon in epsilon_Vals:
                key = "{},{}".format(C, epsilon)
                linear_local[key] = []
                selected_features_local[key] = None

        X_train, X_test, y_train, y_test = None, None, None, None
        if(selector is None and extractor is None):
            X_train, X_test, y_train, y_test = prepare_function(df)
        for C in C_Vals:
            for epsilon in epsilon_Vals:
                selected_features = None
                if(selector is not None):
                    estimator = SVR(kernel='linear', C=C, epsilon=epsilon, random_state=67)
                    X_train, X_test, y_train, y_test, selected_features, current_feature_names = prepare_function(df, estimator)
                    feature_names_local = list(current_feature_names)
                elif(extractor is not None):
                    X_train, X_test, y_train, y_test = prepare_function(df, k,'linear')

                print(f"Training linear kernel SVR with C={C} and epsilon={epsilon} for player {df['player'].iloc[0]}")
                acc = train_linear_kernel_svr(X_train, X_test, y_train, y_test, C, epsilon)
                key = "{},{}".format(C, epsilon)
                linear_local[key].append(acc)
                selected_features_local[key] = selected_features

        return linear_local, selected_features_local, feature_names_local

    results = Parallel(n_jobs=-1)(delayed(process_player)(df) for df in dfs)

    for linear_local, selected_features_local, feature_names_local in results:
        if feature_names is None and feature_names_local is not None:
            feature_names = feature_names_local
            feature_name_to_index = {name: i for i, name in enumerate(feature_names)}
            for existing_key in linear_scores.keys():
                selected_features_dict[existing_key] = [0] * len(feature_names)
        for key in linear_local.keys():
            for acc in linear_local[key]:
                bisect.insort(linear_scores[key], acc)
                if acc > max_linear[key]:
                    max_linear[key] = acc
            selected_features = selected_features_local[key]
            if selected_features is not None:
                for feature in selected_features:
                    selected_features_dict[key][feature_name_to_index[feature]] += 1
                
    with open(file_prefix+'svr_linear_scores.csv', 'w') as f:
        f.write('C,epsilon,avg_score,median_score,max_score,selected_features,feature_names\n')
        for key in linear_scores.keys():
            C, epsilon = key.split(',')
            avg_score = np.mean(linear_scores[key])
            median_score = np.median(linear_scores[key])
            max_score = max_linear[key]
            selected_features = selected_features_dict.get(key, None)
            if selected_features is not None:
                selected_features = [count / len(dfs) for count in selected_features]
            f.write(f"{C},{epsilon},{avg_score},{median_score},{max_score},{selected_features},{feature_names}\n")

def train_and_score_poly_kernel_svr(df, prepare_function, file_prefix, selector=None, extractor=None, k=10):
    dfs = separate_players(df)
    C_Vals = [0.1, 1, 10]
    epsilon_Vals = [0.01, 0.1, 1]
    degree_Vals = [2, 3, 4]
    poly_scores = {}
    max_poly = {}
    selected_features_dict = {}
    selected_features = None
    feature_names = None
    feature_name_to_index = None

    if selector is not None and extractor is not None:
        raise Exception("Both feature selection and feature extraction are selected. Both cannot be done")

    for C in C_Vals:
        for epsilon in epsilon_Vals:
            for degree in degree_Vals:
                poly_key = "{},{},{}".format(degree, C, epsilon)
                poly_scores[poly_key] = []
                max_poly[poly_key] = 0

    def process_player(df):
        poly_local = {}
        selected_features_local = {}
        feature_names_local = None

        for C in C_Vals:
            for epsilon in epsilon_Vals:
                for degree in degree_Vals:
                    poly_key = "{},{},{}".format(degree, C, epsilon)
                    poly_local[poly_key] = []
                    selected_features_local[poly_key] = None

        X_train, X_test, y_train, y_test = None, None, None, None
        if(selector is None and extractor is None):
            X_train, X_test, y_train, y_test = prepare_function(df)
        for C in C_Vals:
            for epsilon in epsilon_Vals:
                for degree in degree_Vals:
                    selected_features = None
                    poly_key = "{},{},{}".format(degree, C, epsilon)
                    if(selector is not None):
                        estimator = SVR(kernel='linear', degree=degree, C=C, epsilon=epsilon, random_state=67)
                        X_train, X_test, y_train, y_test, selected_features, current_feature_names = prepare_function(df, estimator)
                        feature_names_local = list(current_feature_names)
                    elif(extractor is not None):
                        X_train, X_test, y_train, y_test = prepare_function(df, k, 'poly')

                    print(f"Training polynomial kernel SVR with degree={degree}, C={C} and epsilon={epsilon} for player {df['player'].iloc[0]}")
                    acc = train_poly_kernel_svr(X_train, X_test, y_train, y_test, degree, C, epsilon)
                    poly_local[poly_key].append(acc)
                    selected_features_local[poly_key] = selected_features

        return poly_local, selected_features_local, feature_names_local

    results = Parallel(n_jobs=-1)(delayed(process_player)(df) for df in dfs)

    for poly_local, selected_features_local, feature_names_local in results:
        if feature_names is None and feature_names_local is not None:
            feature_names = feature_names_local
            feature_name_to_index = {name: i for i, name in enumerate(feature_names)}
            for existing_key in poly_scores.keys():
                selected_features_dict[existing_key] = [0] * len(feature_names)
        for key in poly_local.keys():
            for acc in poly_local[key]:
                bisect.insort(poly_scores[key], acc)
                if acc > max_poly[key]:
                    max_poly[key] = acc
            selected_features = selected_features_local[key]
            if selected_features is not None:
                for feature in selected_features:
                    selected_features_dict[key][feature_name_to_index[feature]] += 1

    with open(file_prefix+'svr_poly_scores.csv', 'w') as f:
        f.write('degree,C,epsilon,avg_score,median_score,max_score,selected_features,feature_names\n')
        for key in poly_scores.keys():
            degree, C, epsilon = key.split(',')
            avg_score = np.mean(poly_scores[key])
            median_score = np.median(poly_scores[key])
            max_score = max_poly[key]
            selected_features = selected_features_dict.get(key, None)
            if selected_features is not None:
                selected_features = [count / len(dfs) for count in selected_features]
            f.write(f"{degree},{C},{epsilon},{avg_score},{median_score},{max_score},{selected_features},{feature_names}\n")

def train_and_score_rbf_kernel_svr(df, prepare_function, file_prefix, selector=None, extractor=None, k=10):
    dfs = separate_players(df)
    C_Vals = [0.1, 1, 10]
    epsilon_Vals = [0.01, 0.1, 1]
    gamma_Vals = ['scale', 'auto']
    rbf_scores = {}
    max_rbf = {}
    selected_features_dict = {}
    selected_features = None
    feature_names = None
    feature_name_to_index = None

    if selector is not None and extractor is not None:
        raise Exception("Both feature selection and feature extraction are selected. Both cannot be done")

    for C in C_Vals:
        for epsilon in epsilon_Vals:
            for gamma in gamma_Vals:
                rbf_key = "{},{},{}".format(gamma, C, epsilon)
                rbf_scores[rbf_key] = []
                max_rbf[rbf_key] = 0

    def process_player(df):
        rbf_local = {}
        selected_features_local = {}
        feature_names_local = None

        for C in C_Vals:
            for epsilon in epsilon_Vals:
                for gamma in gamma_Vals:
                    rbf_key = "{},{},{}".format(gamma, C, epsilon)
                    rbf_local[rbf_key] = []
                    selected_features_local[rbf_key] = None

        X_train, X_test, y_train, y_test = None, None, None, None
        if(selector is None and extractor is None):
            X_train, X_test, y_train, y_test = prepare_function(df)
        for C in C_Vals:
            for epsilon in epsilon_Vals:
                for gamma in gamma_Vals:
                    selected_features = None
                    rbf_key = "{},{},{}".format(gamma, C, epsilon)
                    if(selector is not None):
                        estimator = SVR(kernel='linear', gamma=gamma, C=C, epsilon=epsilon, random_state=67)
                        X_train, X_test, y_train, y_test, selected_features, current_feature_names = prepare_function(df, estimator)
                        feature_names_local = list(current_feature_names)
                    elif(extractor is not None):
                        X_train, X_test, y_train, y_test = prepare_function(df, k, 'rbf')
                    
                    print(f"Training RBF kernel SVR with gamma={gamma}, C={C} and epsilon={epsilon} for player {df['player'].iloc[0]}")
                    acc = train_rbf_kernel_svr(X_train, X_test, y_train, y_test, gamma, C, epsilon)
                    rbf_local[rbf_key].append(acc)
                    selected_features_local[rbf_key] = selected_features

        return rbf_local, selected_features_local, feature_names_local

    results = Parallel(n_jobs=-1)(delayed(process_player)(df) for df in dfs)

    for rbf_local, selected_features_local, feature_names_local in results:
        if feature_names is None and feature_names_local is not None:
            feature_names = feature_names_local
            feature_name_to_index = {name: i for i, name in enumerate(feature_names)}
            for existing_key in rbf_scores.keys():
                selected_features_dict[existing_key] = [0] * len(feature_names)
        for key in rbf_local.keys():
            for acc in rbf_local[key]:
                bisect.insort(rbf_scores[key], acc)
                if acc > max_rbf[key]:
                    max_rbf[key] = acc
            selected_features = selected_features_local[key]
            if selected_features is not None:
                for feature in selected_features:
                    selected_features_dict[key][feature_name_to_index[feature]] += 1

    with open(file_prefix+'svr_rbf_scores.csv', 'w') as f:
        f.write('gamma,C,epsilon,avg_score,median_score,max_score,selected_features,feature_names\n')
        for key in rbf_scores.keys():
            gamma, C, epsilon = key.split(',')
            avg_score = np.mean(rbf_scores[key])
            median_score = np.median(rbf_scores[key])
            max_score = max_rbf[key]
            selected_features = selected_features_dict.get(key, None)
            if selected_features is not None:
                selected_features = [count / len(dfs) for count in selected_features]
            f.write(f"{gamma},{C},{epsilon},{avg_score},{median_score},{max_score},{selected_features},{feature_names}\n")

def train_and_score_sigmoid_kernel_svr(df, prepare_function, file_prefix, selector=None, extractor=None, k=10):
    dfs = separate_players(df)
    C_Vals = [0.1, 1, 10]
    epsilon_Vals = [0.01, 0.1, 1]
    sigmoid_scores = {}
    max_sigmoid = {}
    selected_features_dict = {}
    selected_features = None
    feature_names = None
    feature_name_to_index = None

    if selector is not None and extractor is not None:
        raise Exception("Both feature selection and feature extraction are selected. Both cannot be done")

    for C in C_Vals:
        for epsilon in epsilon_Vals:
            key = "{},{}".format(C, epsilon)
            sigmoid_scores[key] = []
            max_sigmoid[key] = 0

    def process_player(df):
        sigmoid_local = {}
        selected_features_local = {}
        feature_names_local = None

        for C in C_Vals:
            for epsilon in epsilon_Vals:
                key = "{},{}".format(C, epsilon)
                sigmoid_local[key] = []
                selected_features_local[key] = None

        X_train, X_test, y_train, y_test = None, None, None, None
        if(selector is None and extractor is None):
            X_train, X_test, y_train, y_test = prepare_function(df)
        for C in C_Vals:
            for epsilon in epsilon_Vals:
                selected_features = None
                key = "{},{}".format(C, epsilon)
                if(selector is not None):
                    estimator=SVR(kernel='linear', C=C, epsilon=epsilon, random_state=67)
                    X_train, X_test, y_train, y_test, selected_features, current_feature_names = prepare_function(df, estimator)
                    feature_names_local = list(current_feature_names)
                elif(extractor is not None):
                    X_train, X_test, y_train, y_test = prepare_function(df, k, 'sigmoid')
                
                print(f"Training sigmoid kernel SVR with C={C} and epsilon={epsilon} for player {df['player'].iloc[0]}")
                acc = train_sigmoid_kernel_svr(X_train, X_test, y_train, y_test, C, epsilon)
                sigmoid_local[key].append(acc)
                selected_features_local[key] = selected_features

        return sigmoid_local, selected_features_local, feature_names_local

    results = Parallel(n_jobs=-1)(delayed(process_player)(df) for df in dfs)

    for sigmoid_local, selected_features_local, feature_names_local in results:
        if feature_names is None and feature_names_local is not None:
            feature_names = feature_names_local
            feature_name_to_index = {name: i for i, name in enumerate(feature_names)}
            for existing_key in sigmoid_scores.keys():
                selected_features_dict[existing_key] = [0] * len(feature_names)
        for key in sigmoid_local.keys():
            for acc in sigmoid_local[key]:
                bisect.insort(sigmoid_scores[key], acc)
                if acc > max_sigmoid[key]:
                    max_sigmoid[key] = acc
            selected_features = selected_features_local[key]
            if selected_features is not None:
                for feature in selected_features:
                    selected_features_dict[key][feature_name_to_index[feature]] += 1

    with open(file_prefix+'svr_sigmoid_scores.csv', 'w') as f:
        f.write('C,epsilon,avg_score,median_score,max_score,selected_features,feature_names\n')
        for key in sigmoid_scores.keys():
            C, epsilon = key.split(',')
            avg_score = np.mean(sigmoid_scores[key])
            median_score = np.median(sigmoid_scores[key])
            max_score = max_sigmoid[key]
            selected_features = selected_features_dict.get(key, None)
            if selected_features is not None:
                selected_features = [count / len(dfs) for count in selected_features]
            f.write(f"{C},{epsilon},{avg_score},{median_score},{max_score},{selected_features},{feature_names}\n")