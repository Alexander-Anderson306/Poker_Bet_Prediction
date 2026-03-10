from sklearn.svm import SVR
import bisect
import numpy as np
from sklearn.model_selection import train_test_split
from data_prep import separate_players


#different functions to train and score the different SVR models with different kernels and hyperparameters
def train_linear_kernel_svr(X, y, C, epsilon):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    model = SVR(kernel='linear', C=C, epsilon=epsilon).fit(X_train, y_train)
    return model.score(X_test, y_test)

def train_poly_kernel_svr(X, y, degree, C, epsilon):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    model = SVR(kernel='poly', degree=degree, C=C, epsilon=epsilon).fit(X_train, y_train)
    return model.score(X_test, y_test)

def train_rbf_kernel_svr(X, y, gamma, C, epsilon):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    model = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon).fit(X_train, y_train)
    return model.score(X_test, y_test)

def train_sigmoid_kernel_svr(X, y, C, epsilon):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    model = SVR(kernel='sigmoid', C=C, epsilon=epsilon).fit(X_train, y_train)
    return model.score(X_test, y_test)

def train_and_score_models(df, prepare_function, file_prefix):
    #seperate the players into separate dataframes and prepare the data for the regression models
    dfs = separate_players(df)
    C_Vals = [0.1, 1, 10]
    epsilon_Vals = [0.01, 0.1, 1]
    degree_Vals = [2, 3, 4]
    gamma_Vals = ['scale', 'auto']
    #mean scores for each hyperparameter combination
    linear_scores = {}
    poly_scores = {}
    rbf_scores = {}
    sigmoid_scores = {}
    #max scores for each hyperparameter
    max_linear = {}
    max_poly = {}
    max_rbf = {}
    max_sigmoid = {}

    #initialize the keys for the scores dictionaries
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

    for df in dfs:
        #prepare the data for the regression models based on the prepare_function passed in
        X, y = prepare_function(df)
        for C in C_Vals:
            for epsilon in epsilon_Vals:
                #dict value
                key = "{},{}".format(C, epsilon)
                #test linear kernel
                print(f"Training linear kernel SVR with C={C} and epsilon={epsilon} for player {df['player'].iloc[0]}")
                acc = train_linear_kernel_svr(X, y, C, epsilon)
                bisect.insort(linear_scores[key], acc)
                if acc > max_linear[key]:
                    max_linear[key] = acc

                #test poly kernel
                for degree in degree_Vals:
                    print(f"Training polynomial kernel SVR with degree={degree}, C={C} and epsilon={epsilon} for player {df['player'].iloc[0]}")
                    poly_key = "{},{},{}".format(degree, C, epsilon)
                    acc = train_poly_kernel_svr(X, y, degree, C, epsilon)
                    bisect.insort(poly_scores[poly_key], acc)
                    if acc > max_poly[poly_key]:
                        max_poly[poly_key] = acc

                #test rbf kernel
                for gamma in gamma_Vals:
                    print(f"Training RBF kernel SVR with gamma={gamma}, C={C} and epsilon={epsilon} for player {df['player'].iloc[0]}")
                    rbf_key = "{},{},{}".format(gamma, C, epsilon)
                    acc = train_rbf_kernel_svr(X, y, gamma, C, epsilon)
                    bisect.insort(rbf_scores[rbf_key], acc)
                    if acc > max_rbf[rbf_key]:
                        max_rbf[rbf_key] = acc

                #test sigmoid kernel
                print(f"Training sigmoid kernel SVR with C={C} and epsilon={epsilon} for player {df['player'].iloc[0]}")
                acc = train_sigmoid_kernel_svr(X, y, C, epsilon)
                bisect.insort(sigmoid_scores[key], acc)
                if acc > max_sigmoid[key]:
                    max_sigmoid[key] = acc
        #divide the scores by the number of players to get the average score for each hyperparameter combination
    
    #write the scores to a csv file for analysis
    with open(file_prefix+'_linear_scores.csv', 'w') as f:
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

#funciton to train and score the linear kernel svr model with different hyperparameters
def train_and_score_linear_kernel_svr(df, prepare_function, file_prefix, selector=None, extractor=None, k=10):
    dfs = separate_players(df)
    C_Vals = [0.1, 1, 10]
    epsilon_Vals = [0.01, 0.1, 1]
    linear_scores = {}
    max_linear = {}
    selected_features_dict = {}
    selected_features = None

    if(selector != None and extractor != extractor):
        raise Exception("Both feature selection and feature extraction are selected. Both cannot be done")

    for C in C_Vals:
        for epsilon in epsilon_Vals:
            key = "{},{}".format(C, epsilon)
            linear_scores[key] = []
            max_linear[key] = 0

    for df in dfs:
        X, y
        if(selector == None and extractor == None):
            X, y = prepare_function(df)
        for C in C_Vals:
            for epsilon in epsilon_Vals:
                if(selector != None):
                    #create estimator for the feature selector
                    estimator = SVR(kernel='linear', C=C, epsilon=epsilon)
                    X, y, selected_features = prepare_function(df, estimator)
                elif(extractor != None):
                    X, y = prepare_function(df, 'linear', k)

                print(f"Training linear kernel SVR with C={C} and epsilon={epsilon} for player {df['player'].iloc[0]}")
                acc = train_linear_kernel_svr(X, y, C, epsilon)
                key = "{},{}".format(C, epsilon)
                bisect.insort(linear_scores[key], acc)
                if acc > max_linear[key]:
                    max_linear[key] = acc
                if selected_features != None:
                    selected_features_dict[key] = selected_features
                
    with open(file_prefix+'_linear_scores.csv', 'w') as f:
        f.write('C,epsilon,avg_score,median_score,max_score,selected_features\n')
        for key in linear_scores.keys():
            C, epsilon = key.split(',')
            avg_score = np.mean(linear_scores[key])
            median_score = np.median(linear_scores[key])
            max_score = max_linear[key]
            selected_features = selected_features_dict.get(key, None)
            f.write(f"{C},{epsilon},{avg_score},{median_score},{max_score},{selected_features}\n")

def train_and_score_poly_kernel_svr(df, prepare_function, file_prefix, selector=None, extractor=None, k=10):
    dfs = separate_players(df)
    C_Vals = [0.1, 1, 10]
    epsilon_Vals = [0.01, 0.1, 1]
    degree_Vals = [2, 3, 4]
    poly_scores = {}
    max_poly = {}
    selected_features_dict = {}
    selected_features = None

    if(selector != None and extractor != extractor):
        raise Exception("Both feature selection and feature extraction are selected. Both cannot be done")

    for C in C_Vals:
        for epsilon in epsilon_Vals:
            for degree in degree_Vals:
                poly_key = "{},{},{}".format(degree, C, epsilon)
                poly_scores[poly_key] = []
                max_poly[poly_key] = 0
    for df in dfs:
        X, y
        if(selector == None and extractor == None):
            X, y = prepare_function(df)
        for C in C_Vals:
            for epsilon in epsilon_Vals:
                for degree in degree_Vals:
                    poly_key = "{},{},{}".format(degree, C, epsilon)
                    if(selector != None):
                        #create estimator for the selector
                        estimator = SVR(kernel='poly', degree=degree, C=C, epsilon=epsilon)
                        X, y, selected_features = prepare_function(df, estimator)
                    elif(extractor != None):
                        X, y = prepare_function(df, 'poly', k)

                    print(f"Training polynomial kernel SVR with degree={degree}, C={C} and epsilon={epsilon} for player {df['player'].iloc[0]}")
                    poly_key = "{},{},{}".format(degree, C, epsilon)
                    acc = train_poly_kernel_svr(X, y, degree, C, epsilon)
                    bisect.insort(poly_scores[poly_key], acc)
                    if acc > max_poly[poly_key]:
                        max_poly[poly_key] = acc
                    if selected_features != None:
                        selected_features_dict[poly_key] = selected_features

    with open(file_prefix+'svr_poly_scores.csv', 'w') as f:
        f.write('degree,C,epsilon,avg_score,median_score,max_score,selected_features\n')
        for key in poly_scores.keys():
            degree, C, epsilon = key.split(',')
            avg_score = np.mean(poly_scores[key])
            median_score = np.median(poly_scores[key])
            max_score = max_poly[key]
            selected_features = selected_features_dict.get(key, None)
            f.write(f"{degree},{C},{epsilon},{avg_score},{median_score},{max_score},{selected_features}\n")

def train_and_score_rbf_kernel_svr(df, prepare_function, file_prefix, selector=None, extractor=None, k=10):
    dfs = separate_players(df)
    C_Vals = [0.1, 1, 10]
    epsilon_Vals = [0.01, 0.1, 1]
    gamma_Vals = ['scale', 'auto']
    rbf_scores = {}
    max_rbf = {}
    selected_features_dict = {}
    selected_features = None

    #crash if both selector and extractor are not none
    if(selector != None and extractor != extractor):
        raise Exception("Both feature selection and feature extraction are selected. Both cannot be done")

    for C in C_Vals:
        for epsilon in epsilon_Vals:
            for gamma in gamma_Vals:
                rbf_key = "{},{},{}".format(gamma, C, epsilon)
                rbf_scores[rbf_key] = []
                max_rbf[rbf_key] = 0
    for df in dfs:
        X, y
        if(selector == None and extractor == None):
            X, y = prepare_function(df)
        for C in C_Vals:
            for epsilon in epsilon_Vals:
                for gamma in gamma_Vals:
                    if(selector != None):
                        #create the estimator for the feature selecting
                        estimator = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon)
                        X, y = prepare_function(df, estimator)
                    elif(extractor != None):
                        X, y, selected_features = prepare_function(df, 'rbf', k)
                    
                    print(f"Training RBF kernel SVR with gamma={gamma}, C={C} and epsilon={epsilon} for player {df['player'].iloc[0]}")
                    rbf_key = "{},{},{}".format(gamma, C, epsilon)
                    acc = train_rbf_kernel_svr(X, y, gamma, C, epsilon)
                    bisect.insort(rbf_scores[rbf_key], acc)
                    if acc > max_rbf[rbf_key]:
                        max_rbf[rbf_key] = acc
                    if selected_features != None:
                        selected_features_dict[rbf_key] = selected_features

    with open(file_prefix+'svr_rbf_scores.csv', 'w') as f:
        f.write('gamma,C,epsilon,avg_score,median_score,max_score,selected_features\n')
        for key in rbf_scores.keys():
            gamma, C, epsilon = key.split(',')
            avg_score = np.mean(rbf_scores[key])
            median_score = np.median(rbf_scores[key])
            max_score = max_rbf[key]
            selected_features = selected_features_dict.get(key, None)
            f.write(f"{gamma},{C},{epsilon},{avg_score},{median_score},{max_score},{selected_features}\n")

def train_and_score_sigmoid_kernel_svr(df, prepare_function, file_prefix, selector=None, extractor=None, k=10):
    dfs = separate_players(df)
    C_Vals = [0.1, 1, 10]
    epsilon_Vals = [0.01, 0.1, 1]
    sigmoid_scores = {}
    max_sigmoid = {}
    selected_features_dict = {}
    selected_features = None

    #crash if both selector and extractor are not none
    if(selector != None and extractor != extractor):
        raise Exception("Both feature selection and feature extraction are selected. Both cannot be done")

    for C in C_Vals:
        for epsilon in epsilon_Vals:
            key = "{},{}".format(C, epsilon)
            sigmoid_scores[key] = []
            max_sigmoid[key] = 0
    for df in dfs:
        X, y
        if(selector == None and extractor == None):
            X, y = prepare_function(df)
        for C in C_Vals:
            for epsilon in epsilon_Vals:
                if(selector != None):
                    #create the estimator for this step
                    estimator=SVR(kernel='sigmoid', C=C, epsilon=epsilon)
                    X, y, selected_features = prepare_function(df, estimator)
                elif(extractor != None):
                    X, y = prepare_function(df, k, 'sigmoid')
                
                print(f"Training sigmoid kernel SVR with C={C} and epsilon={epsilon} for player {df['player'].iloc[0]}")
                key = "{},{}".format(C, epsilon)
                acc = train_sigmoid_kernel_svr(X, y, C, epsilon)
                bisect.insort(sigmoid_scores[key], acc)
                if acc > max_sigmoid[key]:
                    max_sigmoid[key] = acc
                if selected_features != None:
                    selected_features_dict[key] = selected_features

    with open(file_prefix+'svr_sigmoid_scores.csv', 'w') as f:
        f.write('C,epsilon,avg_score,median_score,max_score,selected_features\n')
        for key in sigmoid_scores.keys():
            C, epsilon = key.split(',')
            avg_score = np.mean(sigmoid_scores[key])
            median_score = np.median(sigmoid_scores[key])
            max_score = max_sigmoid[key]
            selected_features = selected_features_dict.get(key, None)
            f.write(f"{C},{epsilon},{avg_score},{median_score},{max_score},{selected_features}\n")