from joblib import Parallel, delayed
from SV_models import *
from data_prep import *
from random_forest import *
from MLP_models import *

df = load_class_data('CSVs/poker_data_clustered_FULL.csv')
df = balance_and_limit_samples(df)

print(df.columns)

def run_job(message, func, **kwargs):
    print(message)
    return func(df, **kwargs)

jobs = [
    (
        "Training Player Models for persona classification all SVC Kernels...",
        train_and_score_SVC_models,
        {
            "prepare_function": prepare_persona_predictor_all,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_'
        }
    ),
    (
        "Training Player Models for persona classification Linear Kernel RFECV...",
        train_and_score_linear_kernel_svc,
        {
            "prepare_function": prepare_persona_predictor_RFECV_all,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_RFECV_',
            "selector": 's'
        }
    ),
    (
        "Training Player Models for persona classification Linear Kernel KPCA 10...",
        train_and_score_linear_kernel_svc,
        {
            "prepare_function": prepare_persona_predictor_KPCA_all,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_KPCA_10_',
            "extractor": 'e'
        }
    ),
    (
        "Training Player Models for persona classification Linear Kernel KPCA 15...",
        train_and_score_linear_kernel_svc,
        {
            "prepare_function": prepare_persona_predictor_KPCA_all,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_KPCA_15_',
            "extractor": 'e',
            "k": 15
        }
    ),
    (
        "Training Player Models for persona classification Linear Kernel KPCA 20...",
        train_and_score_linear_kernel_svc,
        {
            "prepare_function": prepare_persona_predictor_KPCA_all,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_KPCA_20_',
            "extractor": 'e',
            "k": 20
        }
    ),
    (
        "Training Player Models for persona classification Random Forest...",
        train_and_score_rf_classifier,
        {
            "prepare_function": prepare_persona_predictor_all,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_'
        }
    ),
    (
        "Training Player Models for persona classification Random Forest RFECV...",
        train_and_score_rf_classifier,
        {
            "prepare_function": prepare_persona_predictor_RFECV_all,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_RFECV_',
            "selector": 's'
        }
    ),
    (
        "Training Player Models for persona classification Random Forest KPCA 10...",
        train_and_score_rf_classifier,
        {
            "prepare_function": prepare_persona_predictor_KPCA_all,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_KPCA_10_',
            "extractor": 'e'
        }
    ),
    (
        "Training Player Models for persona classification Random Forest KPCA 15...",
        train_and_score_rf_classifier,
        {
            "prepare_function": prepare_persona_predictor_KPCA_all,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_KPCA_15_',
            "extractor": 'e',
            "k": 15
        }
    ),
    (
        "Training Player Models for persona classification Random Forest KPCA 20...",
        train_and_score_rf_classifier,
        {
            "prepare_function": prepare_persona_predictor_KPCA_all,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_KPCA_20_',
            "extractor": 'e',
            "k": 20
        }
    ),
    (
        "Training Player Models for persona classification MLP...",
        train_and_score_MLP_classifier,
        {
            "prepare_function": prepare_persona_predictor_all,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_'
        }
    ),
    (
        "Training Player Models for persona classification MLP RFECV...",
        train_and_score_MLP_classifier,
        {
            "prepare_function": prepare_persona_predictor_RFECV_all,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_RFECV_',
            "selector": 's'
        }
    ),
    (
        "Training Player Models for persona classification MLP KPCA 10...",
        train_and_score_MLP_classifier,
        {
            "prepare_function": prepare_persona_predictor_KPCA_all,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_KPCA_10_',
            "extractor": 'e'
        }
    ),
    (
        "Training Player Models for persona classification MLP KPCA 15...",
        train_and_score_MLP_classifier,
        {
            "prepare_function": prepare_persona_predictor_KPCA_all,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_KPCA_15_',
            "extractor": 'e',
            "k": 15
        }
    ),
    (
        "Training Player Models for persona classification MLP KPCA 20...",
        train_and_score_MLP_classifier,
        {
            "prepare_function": prepare_persona_predictor_KPCA_all,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_KPCA_20_',
            "extractor": 'e',
            "k": 20
        }
    ),
    (
        "Training Player Models for persona classification with card info all SVC Kernels...",
        train_and_score_SVC_models,
        {
            "prepare_function": prepare_persona_predictor_card_info,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_card_info_'
        }
    ),
    (
        "Training Player Models for persona classification with card info Linear Kernel RFECV...",
        train_and_score_linear_kernel_svc,
        {
            "prepare_function": prepare_persona_predictor_RFECV_card_info,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_card_info_RFECV_',
            "selector": 's'
        }
    ),
    (
        "Training Player Models for persona classification with card info Linear Kernel KPCA 10...",
        train_and_score_linear_kernel_svc,
        {
            "prepare_function": prepare_persona_predictor_KPCA_card_info,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_card_info_KPCA_10_',
            "extractor": 'e'
        }
    ),
    (
        "Training Player Models for persona classification with card info Linear Kernel KPCA 15...",
        train_and_score_linear_kernel_svc,
        {
            "prepare_function": prepare_persona_predictor_KPCA_card_info,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_card_info_KPCA_15_',
            "extractor": 'e',
            "k": 15
        }
    ),
    (
        "Training Player Models for persona classification with card info Linear Kernel KPCA 20...",
        train_and_score_linear_kernel_svc,
        {
            "prepare_function": prepare_persona_predictor_KPCA_card_info,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_card_info_KPCA_20_',
            "extractor": 'e',
            "k": 20
        }
    ),
    (
        "Training Player Models for persona classification with card info Random Forest...",
        train_and_score_rf_classifier,
        {
            "prepare_function": prepare_persona_predictor_card_info,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_card_info_'
        }
    ),
    (
        "Training Player Models for persona classification with card info Random Forest RFECV...",
        train_and_score_rf_classifier,
        {
            "prepare_function": prepare_persona_predictor_RFECV_card_info,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_card_info_RFECV_',
            "selector": 's'
        }
    ),
    (
        "Training Player Models for persona classification with card info Random Forest KPCA 10...",
        train_and_score_rf_classifier,
        {
            "prepare_function": prepare_persona_predictor_KPCA_card_info,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_card_info_KPCA_10_',
            "extractor": 'e'
        }
    ),
    (
        "Training Player Models for persona classification with card info Random Forest KPCA 15...",
        train_and_score_rf_classifier,
        {
            "prepare_function": prepare_persona_predictor_KPCA_card_info,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_card_info_KPCA_15_',
            "extractor": 'e',
            "k": 15
        }
    ),
    (
        "Training Player Models for persona classification with card info Random Forest KPCA 20...",
        train_and_score_rf_classifier,
        {
            "prepare_function": prepare_persona_predictor_KPCA_card_info,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_card_info_KPCA_20_',
            "extractor": 'e',
            "k": 20
        }
    ),
    (
        "Training Player Models for persona classification with card info MLP...",
        train_and_score_MLP_classifier,
        {
            "prepare_function": prepare_persona_predictor_card_info,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_card_info_'
        }
    ),
    (
        "Training Player Models for persona classification with card info MLP RFECV...",
        train_and_score_MLP_classifier,
        {
            "prepare_function": prepare_persona_predictor_RFECV_card_info,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_card_info_RFECV_',
            "selector": 's'
        }
    ),
    (
        "Training Player Models for persona classification with card info MLP KPCA 10...",
        train_and_score_MLP_classifier,
        {
            "prepare_function": prepare_persona_predictor_KPCA_card_info,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_card_info_KPCA_10_',
            "extractor": 'e'
        }
    ),
    (
        "Training Player Models for persona classification with card info MLP KPCA 15...",
        train_and_score_MLP_classifier,
        {
            "prepare_function": prepare_persona_predictor_KPCA_card_info,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_card_info_KPCA_15_',
            "extractor": 'e',
            "k": 15
        }
    ),
    (
        "Training Player Models for persona classification with card info MLP KPCA 20...",
        train_and_score_MLP_classifier,
        {
            "prepare_function": prepare_persona_predictor_KPCA_card_info,
            "file_prefix": 'CSVs/CLASS_CSVs/persona_predictor_card_info_KPCA_20_',
            "extractor": 'e',
            "k": 20
        }
    ),
]

Parallel(n_jobs=4, backend="threading")(
    delayed(run_job)(message, func, **kwargs)
    for message, func, kwargs in jobs
)