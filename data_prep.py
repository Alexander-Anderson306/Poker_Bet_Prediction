import pandas as pd
import numpy as np
import ast
from sklearn.feature_selection import RFECV
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

#TODO: change this encoding to be more semantic
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
    df.drop(['timestamp', 'month', 'pre_category', 'win_amt'], axis=1, inplace=True)
    #drop the rows where the player has played less than 1500 games to ensure we have at least some data for each player
    player_counts = df['player'].value_counts()
    players_to_keep = player_counts[player_counts >= 1500].index
    df = df[df['player'].isin(players_to_keep)].copy()
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

#function to separate players into separate dataframes based on player id
def separate_players(df):
    player_dfs = []
    for player_id, group in df.groupby('player'):
        player_dfs.append(group)
    return player_dfs
    
############################################################ Bet Predictor Preprocessing ############################################################
def prepare_bet_predictor_data(df):
    X = df.drop(columns=['player', 'bet_total'])
    y = df['bet_total'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def prepare_bet_predictor_RFECV(df, estimator):
    X = df.drop(columns=['player', 'bet_total'])
    feature_names = X.columns
    y = df['bet_total'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    selector = RFECV(estimator=estimator, step=1, cv=5, scoring='neg_mean_squared_error')

    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)
    selected_features = feature_names[selector.get_support()]
    return X_train, X_test, y_train, y_test, selected_features, feature_names

def prepare_bet_predictor_KPCA(df, k, kernel):
    X = df.drop(columns=['player', 'bet_total'])
    y = df['bet_total'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    kpca = KernelPCA(n_components=k, kernel=kernel)
    X_train = kpca.fit_transform(X_train)
    X_test = kpca.transform(X_test)

    return X_train, X_test, y_train, y_test

############################################################ Card Predictor Preprecessing ############################################################

def prepare_card_predictor_data(df):
    X = df.drop(columns=['player', 'hole_0', 'hole_1', 'flop_strength', 'turn_strength', 'river_strength'])
    y = df['river_strength'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def prepare_card_predictor_RFECV(df, estimator):
    X = df.drop(columns=['player', 'hole_0', 'hole_1', 'flop_strength', 'turn_strength', 'river_strength'])
    feature_names = X.columns
    y = df['river_strength'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    selector = RFECV(estimator=estimator, step=1, cv=5, scoring='neg_mean_squared_error')

    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)
    selected_features = feature_names[selector.get_support()]
    return X_train, X_test, y_train, y_test, selected_features, feature_names

def prepare_card_predictor_KPCA(df, k, kernel):
    X = df.drop(columns=['player', 'hole_0', 'hole_1', 'flop_strength', 'turn_strength', 'river_strength'])
    y = df['river_strength'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    kpca = KernelPCA(n_components=k, kernel=kernel)
    X_train = kpca.fit_transform(X_train)
    X_test = kpca.transform(X_test)

    return X_train, X_test, y_train, y_test