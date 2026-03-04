import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import ast
import bisect

#function to map the action to semantic values for the regression model
def action_semantic_features(s: str):
    _ALLOWED = set("BkbcrA")
    if any(ch not in _ALLOWED for ch in s):
        return np.zeros(9, dtype=np.float64)

    #blind B can only appear at the start, and all in A can only appear at the end, and there can only be one of each
    if s.count("B") > 1 or ("B" in s and not s.startswith("B")):
        return None
    if s.count("A") > 1 or ("A" in s and not s.endswith("A")):
        return None

    feats = np.array([
        len(s),
        1 if "B" in s else 0,
        1 if "A" in s else 0,
        s.count("k"),
        s.count("b"),
        s.count("c"),
        s.count("r"),
        1 if s.startswith("B") else 0,
        1 if s.endswith("A") else 0,
    ], dtype=np.float64)

    return feats

#function to map the hole cards to semantic values for the regression model
def hole_cards_semantic_features(s: list):
    if len(s) != 2:
        #print(f"Expected 2 hole cards, got {len(s)} for {s!r}")
        return None

    def card_to_features(card):
        rank = card[:-1]
        suit = card[-1]
        rank_value = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9, 'T': 10,
            'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }.get(rank, 0)
        suit_value = {
            'h': 0.25,
            'd': 0.5,
            'c': 0.75,
            's': 1.0
        }.get(suit, 0)
        return rank_value * suit_value

    features = np.array([card_to_features(card) for card in s], dtype=np.float64)
    return features

#function to load the data into a pandas dataframe and preprocess it
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(['timestamp', 'month', 'pre_category', 'is_folded_strong', 'is_bluff', 'is_set_success', 'win_amt'], axis=1, inplace=True)
    #drop the rows where the player has played less than 10 games to ensure we have at least some data for each player
    player_counts = df['player'].value_counts()
    players_to_keep = player_counts[player_counts >= 1000].index
    df = df[df['player'].isin(players_to_keep)]
    df['hole_cards'] = df['hole_cards'].apply(ast.literal_eval)

    #transform the categorical features into semantic features for the regression model
    df['hole_cards'] = df['hole_cards'].map(hole_cards_semantic_features)
    df['flop_act'] = df['flop_act'].map(action_semantic_features)
    df['turn_act'] = df['turn_act'].map(action_semantic_features)
    df['river_act'] = df['river_act'].map(action_semantic_features)

    #drop invalid data
    df.dropna(subset=['hole_cards', 'flop_act', 'turn_act', 'river_act'], inplace=True)

    #convert the values into proper list of floats for the regression model
    hole = pd.DataFrame(df["hole_cards"].tolist(), index=df.index).add_prefix("hole_")
    flop = pd.DataFrame(df["flop_act"].tolist(), index=df.index).add_prefix("flop_")
    turn = pd.DataFrame(df["turn_act"].tolist(), index=df.index).add_prefix("turn_")
    river = pd.DataFrame(df["river_act"].tolist(), index=df.index).add_prefix("river_")

    df = pd.concat(
        [df.drop(columns=["hole_cards","flop_act","turn_act","river_act"]),
        hole, flop, turn, river],
        axis=1
    )
    return df


#print the avg number of games each player has played
def print_avg_games_per_player(df):
    player_counts = df['player'].value_counts()
    avg_games = player_counts.mean()
    print(f"Average number of games per player: {avg_games:.2f}")


#function to separate players into separate dataframes based on player id
def separate_players(df):
    player_dfs = []
    for player_id, group in df.groupby('player'):
        player_dfs.append(group)
    return player_dfs

#function to split the data into scaled features and targets for the regression models
def prepare_data_for_model(df):
    X = df.drop(columns=['player', 'bet_total'])
    y = df['bet_total'].to_numpy()
    X = StandardScaler().fit_transform(X)
    return X, y

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

def train_and_score_all_models(df):
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
        X, y = prepare_data_for_model(df)
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
    with open('svr_linear_scores.csv', 'w') as f:
        f.write('C,epsilon,avg_score,median_score,max_score\n')
        for key in linear_scores.keys():
            C, epsilon = key.split(',')
            avg_score = np.mean(linear_scores[key])
            median_score = np.median(linear_scores[key])
            max_score = max_linear[key]
            f.write(f"{C},{epsilon},{avg_score},{median_score},{max_score}\n")
    
    with open('svr_poly_scores.csv', 'w') as f:
        f.write('degree,C,epsilon,avg_score,median_score,max_score\n')
        for key in poly_scores.keys():
            degree, C, epsilon = key.split(',')
            avg_score = np.mean(poly_scores[key])
            median_score = np.median(poly_scores[key])
            max_score = max_poly[key]
            f.write(f"{degree},{C},{epsilon},{avg_score},{median_score},{max_score}\n")
    
    with open('svr_rbf_scores.csv', 'w') as f:
        f.write('gamma,C,epsilon,avg_score,median_score,max_score\n')
        for key in rbf_scores.keys():
            gamma, C, epsilon = key.split(',')
            avg_score = np.mean(rbf_scores[key])
            median_score = np.median(rbf_scores[key])
            max_score = max_rbf[key]
            f.write(f"{gamma},{C},{epsilon},{avg_score},{median_score},{max_score}\n")
    
    with open('svr_sigmoid_scores.csv', 'w') as f:
        f.write('C,epsilon,avg_score,median_score,max_score\n')
        for key in sigmoid_scores.keys():
            C, epsilon = key.split(',')
            avg_score = np.mean(sigmoid_scores[key])
            median_score = np.median(sigmoid_scores[key])
            max_score = max_sigmoid[key]
            f.write(f"{C},{epsilon},{avg_score},{median_score},{max_score}\n")


df = load_data('poker_analysis_results.csv')
dfs = separate_players(df)
train_and_score_all_models(df)