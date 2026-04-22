from data_prep import load_data
from SV_models import *
from data_prep import *
from random_forest import *
from MLP_models import *

df = load_data('CSVs/combined_with_persona.csv') 

print(df.columns)

#train without feature selection
print("Training Player Models for bet prediction all SVR Kernels...")
train_and_score_models(df, prepare_function=prepare_bet_predictor_data, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_')
print("Training Player Models for card prediction all SVR Kernels...")
train_and_score_models(df, prepare_function=prepare_card_predictor_data, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_')

#train with feature selection
print("Training Player Models for bet prediction Linear Kernel RFECV...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_bet_predictor_RFECV, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_RFECV_', selector='s')
print("Training Player Models for card prediction Linear Kernel RFECV...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_card_predictor_RFECV, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_RFECV_', selector='s')

#train with feature extraction with num features = 10
print("Training Player Models for bet prediction Linear Kernel KPCA 10...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_10_', extractor='e')
print("Training Player Models for card prediction Linear Kernel KPCA 10...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_10_', extractor='e')

#train with feature extraction with num features = 15
print("Training Player Models for bet prediction Linear Kernel KPCA 15...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_15_', extractor='e', k=15)
print("Training Player Models for card prediction Linear Kernel KPCA 15...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_15_', extractor='e', k=15)

#train with feature extraction with num features = 20
print("Training Player Models for bet prediction Linear Kernel KPCA 20...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_20_', extractor='e', k=20)
print("Training Player Models for card prediction Linear Kernel KPCA 20...")
train_and_score_linear_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_20_', extractor='e', k=20)

#now train the trees
print("Training Player Models for bet prediction Random Forest...")
train_and_score_rf(df, prepare_function=prepare_bet_predictor_data, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_')
print("Training Player Models for card prediction Random Forest...")
train_and_score_rf(df, prepare_function=prepare_card_predictor_data, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_')

print("Training Player Models for bet prediction Random Forest RFECV...")
train_and_score_rf(df, prepare_function=prepare_bet_predictor_RFECV, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_RFECV_', selector='s')
print("Training Player Models for card prediction Random Forest RFECV...")
train_and_score_rf(df, prepare_function=prepare_card_predictor_RFECV, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_RFECV_', selector='s')

print("Training Player Models for bet prediction Random Forest KPCA 10...")
train_and_score_rf(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_10_', extractor='e')
print("Training Player Models for card prediction Random Forest KPCA 10...")
train_and_score_rf(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_10_', extractor='e')

print("Training Player Models for bet prediction Random Forest KPCA 15...")
train_and_score_rf(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_15_', extractor='e', k=15)
print("Training Player Models for card prediction Random Forest KPCA 15...")
train_and_score_rf(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_15_', extractor='e', k=15)

print("Training Player Models for bet prediction Random Forest KPCA 20...")
train_and_score_rf(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_20_', extractor='e', k=20)
print("Training Player Models for card prediction Random Forest KPCA 20...")
train_and_score_rf(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_20_', extractor='e', k=20)

#now train the mlps
print("Training Player Models for bet prediction MLP...")
train_and_score_MLP(df, prepare_function=prepare_bet_predictor_data, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_')
print("Training Player Models for card prediction MLP...")
train_and_score_MLP(df, prepare_function=prepare_card_predictor_data, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_')

print("Training Player Models for bet prediction MLP RFECV...")
train_and_score_MLP(df, prepare_function=prepare_bet_predictor_RFECV, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_RFECV_', selector='s')
print("Training Player Models for card prediction MLP RFECV...")
train_and_score_MLP(df, prepare_function=prepare_card_predictor_RFECV, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_RFECV_', selector='s')

print("Training Player Models for bet prediction MLP KPCA 10...")
train_and_score_MLP(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_10_', extractor='e')
print("Training Player Models for card prediction MLP KPCA 10...")
train_and_score_MLP(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_10_', extractor='e')

print("Training Player Models for bet prediction MLP KPCA 15...")
train_and_score_MLP(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_15_', extractor='e', k=15)
print("Training Player Models for card prediction MLP KPCA 15...")
train_and_score_MLP(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_15_', extractor='e', k=15)

print("Training Player Models for bet prediction MLP KPCA 20...")
train_and_score_MLP(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/bet_predictor_KPCA_20_', extractor='e', k=20)
print("Training Player Models for card prediction MLP KPCA 20...")
train_and_score_MLP(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSVs/CLUSTER_CSVs/card_predictor_KPCA_20_', extractor='e', k=20)