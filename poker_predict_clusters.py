from data_prep import load_data
from SVR_models import *
from data_prep import *
from random_forest import *
from MLP_models import *


df = load_data_clustered('CSVs/poker_data_master.csv')

#train without feature selection
print("Training Clustered Models for bet prediction all SVR Kernels...")
train_and_score_models(df, prepare_function=prepare_bet_predictor_data_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_', cluster=True)
print("Training Clustered Models for card prediction all SVR Kernels...")
train_and_score_models(df, prepare_function=prepare_card_predictor_data_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_', cluster=True)

#train with feature selection
print("Training Clustered Models for bet prediction Linear Kernel RFECV...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_bet_predictor_RFECV_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_RFECV_', selector='s', cluster=True)
print("Training Clustered Models for card prediction Linear Kernel RFECV...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_card_predictor_RFECV, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_RFECV_', selector='s', cluster=True)

#train with feature extraction with num features = 10
print("Training Clustered Models for bet prediction Linear Kernel KPCA 10...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_10_', extractor='e', cluster=True)
print("Training Clustered Models for card prediction Linear Kernel KPCA 10...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_10_', extractor='e', cluster=True)


#train with feature extraction with num features = 15
print("Training Clustered Models for bet prediction Linear Kernel KPCA 15...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_15_', extractor='e', k=15, cluster=True)
print("Training Clustered Models for card prediction Linear Kernel KPCA 15...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_15_', extractor='e', k=15, cluster=True)

#train with feature extraction with num features = 20
print("Training Clustered Models for bet prediction Linear Kernel KPCA 20...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_20_', extractor='e', k=20, cluster=True)
print("Training Clustered Models for card prediction Linear Kernel KPCA 20...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_20_', extractor='e', k=20, cluster=True)

#train without feature selection
print("Training Clustered Models for bet prediction all SVR Kernels...")
train_and_score_models(df, prepare_function=prepare_bet_predictor_data_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_', cluster=True)
print("Training Clustered Models for card prediction all SVR Kernels...")
train_and_score_models(df, prepare_function=prepare_card_predictor_data_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_', cluster=True)

#train with feature selection
print("Training Clustered Models for bet prediction Linear Kernel RFECV...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_bet_predictor_RFECV_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_RFECV_', selector='s', cluster=True)
print("Training Clustered Models for card prediction Linear Kernel RFECV...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_card_predictor_RFECV, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_RFECV_', selector='s', cluster=True)

#train with feature extraction with num features = 10
print("Training Clustered Models for bet prediction Linear Kernel KPCA 10...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_10_', extractor='e', cluster=True)
print("Training Clustered Models for card prediction Linear Kernel KPCA 10...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_10_', extractor='e', cluster=True)


#train with feature extraction with num features = 15
print("Training Clustered Models for bet prediction Linear Kernel KPCA 15...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_15_', extractor='e', k=15, cluster=True)
print("Training Clustered Models for card prediction Linear Kernel KPCA 15...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_15_', extractor='e', k=15, cluster=True)

#train with feature extraction with num features = 20
print("Training Clustered Models for bet prediction Linear Kernel KPCA 20...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_20_', extractor='e', k=20, cluster=True)
print("Training Clustered Models for card prediction Linear Kernel KPCA 20...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_20_', extractor='e', k=20, cluster=True)

#now train the trees
train_and_score_rf(df, prepare_function=prepare_bet_predictor_data_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_', cluster=True)
train_and_score_rf(df, prepare_function=prepare_card_predictor_data_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_', cluster=True)

train_and_score_rf(df, prepare_function=prepare_bet_predictor_RFECV_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_RFECV_', selector='s', cluster=True)
train_and_score_rf(df, prepare_function=prepare_card_predictor_RFECV_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_RFECV_', selector='s', cluster=True)

train_and_score_rf(df, prepare_function=prepare_bet_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_10_', extractor='e', cluster=True)
train_and_score_rf(df, prepare_function=prepare_card_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_10_', extractor='e', cluster=True)

train_and_score_rf(df, prepare_function=prepare_bet_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_15_', extractor='e', k=15, cluster=True)
train_and_score_rf(df, prepare_function=prepare_card_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_15_', extractor='e', k=15, cluster=True)

train_and_score_rf(df, prepare_function=prepare_bet_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_20_', extractor='e', k=20, cluster=True)
train_and_score_rf(df, prepare_function=prepare_card_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_20_', extractor='e', k=20, cluster=True)

#now train the mlps
train_and_score_MLP(df, prepare_function=prepare_bet_predictor_data_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_', cluster=True)
train_and_score_MLP(df, prepare_function=prepare_card_predictor_data_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_', cluster=True)

train_and_score_MLP(df, prepare_function=prepare_bet_predictor_RFECV_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_RFECV_', selector='s', cluster=True)
train_and_score_MLP(df, prepare_function=prepare_card_predictor_RFECV_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_RFECV_', selector='s', cluster=True)

train_and_score_MLP(df, prepare_function=prepare_bet_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_10_', extractor='e', cluster=True)
train_and_score_MLP(df, prepare_function=prepare_card_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_10_', extractor='e', cluster=True)

train_and_score_MLP(df, prepare_function=prepare_bet_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_15_', extractor='e', k=15, cluster=True)
train_and_score_MLP(df, prepare_function=prepare_card_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_15_', extractor='e', k=15, cluster=True)

train_and_score_MLP(df, prepare_function=prepare_bet_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_20_', extractor='e', k=20, cluster=True)
train_and_score_MLP(df, prepare_function=prepare_card_predictor_KPCA_clustered, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_20_', extractor='e', k=20, cluster=True)
