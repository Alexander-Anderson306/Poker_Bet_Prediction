from data_prep import load_data
from SVR_models import *
from data_prep import *


df = load_data('CSVs/combined.csv')
#train without feature selection
train_and_score_models(df, prepare_function=prepare_bet_predictor_data, file_prefix='CSV/bet_predictor_')
train_and_score_models(df, prepare_function=prepare_card_predictor_data, file_prefix='CSV/card_predictor_')

#train with feature selection
train_and_score_linear_kernel_svr(df, prepare_function=prepare_bet_predictor_RFECV, file_prefix='CSV/bet_predictor_RFECV_', selector='s')
train_and_score_linear_kernel_svr(df, prepare_function=prepare_card_predictor_RFECV, file_prefix='CSV/card_predictor_RFECV_', selector='s')

train_and_score_poly_kernel_svr(df, prepare_function=prepare_bet_predictor_RFECV, file_prefix='CSV/bet_predictor_RFECV_', selector='s')
train_and_score_poly_kernel_svr(df, prepare_function=prepare_card_predictor_RFECV, file_prefix='CSV/card_predictor_RFECV_', selector='s')

train_and_score_rbf_kernel_svr(df, prepare_function=prepare_bet_predictor_RFECV, file_prefix='CSV/bet_predictor_RFECV_', selector='s')
train_and_score_rbf_kernel_svr(df, prepare_function=prepare_card_predictor_RFECV, file_prefix='CSV/card_predictor_RFECV_', selector='s')

train_and_score_sigmoid_kernel_svr(df, prepare_function=prepare_bet_predictor_RFECV, file_prefix='CSV/bet_predictor_RFECV_', selector='s')
train_and_score_sigmoid_kernel_svr(df, prepare_function=prepare_card_predictor_RFECV, file_prefix='CSV/card_predictor_RFECV_', selector='s')

#train with feature extraction with num features = 10
train_and_score_linear_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSV/bet_predictor_KPCA_10_', extractor='e')
train_and_score_linear_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSV/card_predictor_KPCA_10_', extractor='e')

train_and_score_poly_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSV/bet_predictor_KPCA_10_', extractor='e')
train_and_score_poly_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSV/card_predictor_KPCA_10_', extractor='e')

train_and_score_rbf_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSV/bet_predictor_KPCA_10_', extractor='e')
train_and_score_rbf_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSV/card_predictor_KPCA_10_', extractor='e')

train_and_score_sigmoid_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSV/bet_predictor_KPCA_10_', extractor='e')
train_and_score_sigmoid_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSV/card_predictor_KPCA_10_', extractor='e')

#train with feature extraction with num features = 15
train_and_score_linear_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSV/bet_predictor_KPCA_15_', extractor='e', k=15)
train_and_score_linear_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSV/card_predictor_KPCA_15_', extractor='e', k=15)

train_and_score_poly_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSV/bet_predictor_KPCA_15_', extractor='e', k=15)
train_and_score_poly_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSV/card_predictor_KPCA_15_', extractor='e', k=15)

train_and_score_rbf_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSV/bet_predictor_KPCA_15_', extractor='e', k=15)
train_and_score_rbf_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSV/card_predictor_KPCA_15_', extractor='e', k=15)

train_and_score_sigmoid_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSV/bet_predictor_KPCA_15_', extractor='e', k=15)
train_and_score_sigmoid_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSV/card_predictor_KPCA_15_', extractor='e', k=15)

#train with feature extraction with num features = 20
train_and_score_linear_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSV/bet_predictor_KPCA_20_', extractor='e', k=20)
train_and_score_linear_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSV/card_predictor_KPCA_20_', extractor='e', k=20)

train_and_score_poly_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSV/bet_predictor_KPCA_20_', extractor='e', k=20)
train_and_score_poly_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSV/card_predictor_KPCA_20_', extractor='e', k=20)

train_and_score_rbf_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSV/bet_predictor_KPCA_20_', extractor='e', k=20)
train_and_score_rbf_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSV/card_predictor_KPCA_20_', extractor='e', k=20)

train_and_score_sigmoid_kernel_svr(df, prepare_function=prepare_bet_predictor_KPCA, file_prefix='CSV/bet_predictor_KPCA_20_', extractor='e', k=20)
train_and_score_sigmoid_kernel_svr(df, prepare_function=prepare_card_predictor_KPCA, file_prefix='CSV/card_predictor_KPCA_20_', extractor='e', k=20)

