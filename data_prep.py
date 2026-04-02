import pandas as pd
import numpy as np
import ast
from sklearn.feature_selection import RFECV
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def action_semantic_features(s: str):
    """
    Encodes the different acts of a poker game semantically.
    We count the number of times each action occures and return that as a numpy array.
    We return None if we get an impossible combination I.E all in twice? Or Small blind twice?
    """
    _ALLOWED = set("BkbcrA")
    if any(ch not in _ALLOWED for ch in s):
        return np.zeros(5, dtype=np.float64)

    feats = np.array([
        len(s),
        s.count("k"),
        s.count("b"),
        s.count("c"),
        s.count("r"),
    ], dtype=np.float64)

    return feats

#function to map the hole cards to semantic values for the regression model
def hole_cards_to_features(cards):
    """
    Mapps the starting cards of a poker game semantically.
    We take a list of two strings which look like CardNumSuit
    I.E 2h for a two of hearts or As for an ace of spades
    We sepearte the number from the suit and encode each as seperate features.
    """
    rank_map = {
        '2':2,'3':3,'4':4,'5':5,'6':6,
        '7':7,'8':8,'9':9,'T':10,
        'J':11,'Q':12,'K':13,'A':14
    }
    suit_map = {'h':0,'d':1,'c':2,'s':3}

    if len(cards) != 2:
        return [None, None, None, None]

    card1, card2 = cards

    rank1 = rank_map[card1[:-1]]
    suit1 = suit_map[card1[-1]]
    rank2 = rank_map[card2[:-1]]
    suit2 = suit_map[card2[-1]]

    return [rank1, suit1, rank2, suit2]

#function to load the data into a pandas dataframe and preprocess it
def load_data(file_path):
    """
    This function loads the csv data into a pandas dataframe.
    We load the data and drop unused columns. We then drop
    all players who played less than 3000 games.
    We also convert features like the starting cards and the game acts into 
    new features.
    """
    df = pd.read_csv(file_path)
    df.drop(['timestamp', 'month', 'pre_category', 'win_amt'], axis=1, inplace=True)
    #drop the rows where the player has played less than 3000 games to ensure we have at least some data for each player
    player_counts = df['player'].value_counts()
    players_to_keep = player_counts[player_counts >= 3000].index
    df = df[df['player'].isin(players_to_keep)].copy()
    df['hole_cards'] = df['hole_cards'].apply(ast.literal_eval)

    #transform the categorical features into semantic features for the regression model
    df[['rank1','suit1','rank2','suit2']] = pd.DataFrame(
    df['hole_cards'].apply(hole_cards_to_features).tolist(),index=df.index)
    df['flop_act'] = df['flop_act'].map(action_semantic_features)
    df['turn_act'] = df['turn_act'].map(action_semantic_features)
    df['river_act'] = df['river_act'].map(action_semantic_features)

    #drop invalid data
    df.dropna(subset=['hole_cards', 'flop_act', 'turn_act', 'river_act'], inplace=True)

    #convert the values into proper list of floats for the regression model
    flop = pd.DataFrame(df["flop_act"].tolist(), index=df.index).add_prefix("flop_")
    turn = pd.DataFrame(df["turn_act"].tolist(), index=df.index).add_prefix("turn_")
    river = pd.DataFrame(df["river_act"].tolist(), index=df.index).add_prefix("river_")

    df = pd.concat(
        [df.drop(columns=["hole_cards","flop_act","turn_act","river_act"]), flop, turn, river], axis=1)
    return df

#function to separate players into separate dataframes based on player id
def separate_players(df):
    """
    Separates the loaded data frame into a list of dataframes for each player.
    """
    player_dfs = []
    for player_id, group in df.groupby('player'):
        player_dfs.append(group)
    return player_dfs
    
############################################################ Bet Predictor Preprocessing ############################################################
def prepare_bet_predictor_data(df):
    """
    Prepares the player by normalizing their feature data, then seperates data into training and testing sets.
    """
    X = df.drop(columns=['player', 'bet_total'])
    y = df['bet_total'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def prepare_bet_predictor_RFECV(df, estimator):
    """
    Like the above prepare function. However we also apply RFECV feature seleciton to select which features we use.
    We return a training set, testing set, and what features we used.
    """
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
    """
    Like the first prepare funciton. However we also apply KernelPCA feature extraction to extract new features and reduce dimensionality.
    """
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
    """
    Prepares the player by normalizing their feature data, then seperates data into training and testing sets.
    """
    X = df.drop(columns=['player', 'rank1','suit1','rank2','suit2', 'flop_strength', 'turn_strength', 'river_strength'])
    y = df['river_strength'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def prepare_card_predictor_RFECV(df, estimator):
    """
    Like the above prepare function. However we also apply RFECV feature seleciton to select which features we use.
    We return a training set, testing set, and what features we used.
    """
    X = df.drop(columns=['player', 'rank1','suit1','rank2','suit2', 'flop_strength', 'turn_strength', 'river_strength'])
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
    """
    Like the first prepare funciton. However we also apply KernelPCA feature extraction to extract new features and reduce dimensionality.
    """
    X = df.drop(columns=['player', 'rank1','suit1','rank2','suit2', 'flop_strength', 'turn_strength', 'river_strength'])
    y = df['river_strength'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    kpca = KernelPCA(n_components=k, kernel=kernel)
    X_train = kpca.fit_transform(X_train)
    X_test = kpca.transform(X_test)

    return X_train, X_test, y_train, y_test

############################################################ Below are the preprocessing functions for the two cluster types ############################################################
def load_data_clustered(file_path):
    df = pd.read_csv(file_path)
    df.drop(['timestamp', 'month', 'pre_category', 'win_amt', 'net_profit', 'flop_bluff_idx', 'turn_bluff_idx', 'river_bluff_idx'], axis=1, inplace=True)
    df['hole_cards'] = df['hole_cards'].apply(ast.literal_eval)

    #transform the categorical features into semantic features for the regression model
    df[['rank1','suit1','rank2','suit2']] = pd.DataFrame(
    df['hole_cards'].apply(hole_cards_to_features).tolist(),index=df.index)
    df['flop_act'] = df['flop_act'].map(action_semantic_features)
    df['turn_act'] = df['turn_act'].map(action_semantic_features)
    df['river_act'] = df['river_act'].map(action_semantic_features)

    #drop invalid data
    df.dropna(subset=['hole_cards', 'flop_act', 'turn_act', 'river_act'], inplace=True)

    #convert the values into proper list of floats for the regression model
    flop = pd.DataFrame(df["flop_act"].tolist(), index=df.index).add_prefix("flop_")
    turn = pd.DataFrame(df["turn_act"].tolist(), index=df.index).add_prefix("turn_")
    river = pd.DataFrame(df["river_act"].tolist(), index=df.index).add_prefix("river_")

    df = pd.concat(
        [df.drop(columns=["hole_cards","flop_act","turn_act","river_act"]), flop, turn, river], axis=1)
    return df

#function to seperate the two clusters
def separate_clusters(df):
    """
    Separates the loaded data frame into a list of dataframes for each cluster.
    """
    cluster_df = []
    for cluster_id, group in df.groupby('persona'):
        cluster_df.append(group)
    return cluster_df
    
############################################################ Bet Predictor Preprocessing ############################################################
def prepare_bet_predictor_data_clustered(df):
    """
    Prepares the cluster by normalizing their feature data, then seperates data into training and testing sets.
    """
    X = df.drop(columns=['persona', 'bet_total', 'player'])
    y = df['bet_total'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def prepare_bet_predictor_RFECV_clustered(df, estimator):
    """
    Like the above prepare function. However we also apply RFECV feature seleciton to select which features we use.
    We return a training set, testing set, and what features we used.
    """
    X = df.drop(columns=['persona', 'bet_total', 'player'])
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

def prepare_bet_predictor_KPCA_clustered(df, k, kernel):
    """
    Like the first prepare funciton. However we also apply KernelPCA feature extraction to extract new features and reduce dimensionality.
    """
    X = df.drop(columns=['persona', 'bet_total', 'player'])
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

def prepare_card_predictor_data_clustered(df):
    """
    Prepares the cluster by normalizing their feature data, then seperates data into training and testing sets.
    """
    X = df.drop(columns=['persona', 'rank1','suit1','rank2','suit2', 'flop_strength', 'turn_strength', 'river_strength', 'player'])
    y = df['river_strength'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def prepare_card_predictor_RFECV_clustered(df, estimator):
    """
    Like the above prepare function. However we also apply RFECV feature seleciton to select which features we use.
    We return a training set, testing set, and what features we used.
    """
    X = df.drop(columns=['persona', 'rank1','suit1','rank2','suit2', 'flop_strength', 'turn_strength', 'river_strength', 'player'])
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

def prepare_card_predictor_KPCA_clustered(df, k, kernel):
    """
    Like the first prepare funciton. However we also apply KernelPCA feature extraction to extract new features and reduce dimensionality.
    """
    X = df.drop(columns=['player', 'rank1','suit1','rank2','suit2', 'flop_strength', 'turn_strength', 'river_strength', 'player'])
    y = df['river_strength'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    kpca = KernelPCA(n_components=k, kernel=kernel)
    X_train = kpca.fit_transform(X_train)
    X_test = kpca.transform(X_test)

    return X_train, X_test, y_train, y_test