# 所有GNN相关的文件路径和参数
import os
from pathlib import Path

# --- Project Root ---
# 此配置文件位于项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# --- Base Directories ---
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Data Subdirectories ---
RAW_DATA_DIR = DATA_DIR / "raw"
MPS_DATA_DIR = DATA_DIR / "mps"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# --- Instance Generation Parameters ---
INSTANCE_GEN = {
    'facilities': {
        'train': {'n_instances': 250, 'dimension': 50, 'ratio': 6, 'overwrite': True}, 
        'valid': {'n_instances': 50, 'dimension': 50, 'ratio': 6, 'overwrite': True}, 
        'test': {'n_instances': 50, 'dimension': 50, 'ratio': 6, 'overwrite': True},  
        # 'transfer_40': {'n_instances': 100, 'dimension': 40, 'ratio': 6, 'overwrite': False},
        # 'transfer_30': {'n_instances': 100, 'dimension': 30, 'ratio': 6, 'overwrite': False},
        # 'transfer_35': {'n_instances': 100, 'dimension': 35, 'ratio': 6, 'overwrite': False},
        # 'transfer_50': {'n_instances': 100, 'dimension': 50, 'ratio': 6, 'overwrite': True},
        # 'transfer_80': {'n_instances': 100, 'dimension': 80, 'ratio': 6, 'overwrite': True},
        # 'transfer_100': {'n_instances': 100, 'dimension': 100, 'ratio': 6, 'overwrite': True},
    },
    'osif': {
        'train': {'n_instances': 250, 'overwrite': True, 'input_dim': 8, 'hidden_dims': [16, 32], 'output_dim': 10},
        'valid': {'n_instances': 50, 'overwrite': True, 'input_dim': 8, 'hidden_dims': [16, 32], 'output_dim': 10},
        'test': {'n_instances': 50, 'overwrite': True, 'input_dim': 8, 'hidden_dims': [16, 32], 'output_dim': 10},
    }
}

# --- Feature Extraction Parameters ---
FEATURE_EXTRACTION = {
    'n_anchors': 256,
}
OSIF_EPSILON = 50

# --- Feature Mode ---
# 'full': structural + betweenness + anchor features
# 'simple': only basic structural features
FEATURES_MODE = 'simple'  # choices: 'full', 'simple'

# --- GNN Model Parameters ---
def _compute_feature_dims():
    n_anchors = FEATURE_EXTRACTION['n_anchors']
    if FEATURES_MODE == 'full':
        # constraints: 6 structural + 1 betweenness + n_anchors
        cons_nfeats = 6 + 1 + n_anchors
        # variables: 5 structural + 1 betweenness + n_anchors
        var_nfeats = 5 + 1 + n_anchors
    else:
        # constraints: [is_le, is_ge, is_eq, degree]
        cons_nfeats = 4
        # variables: [is_binary, is_integer, is_continuous, degree]
        var_nfeats = 4
    return cons_nfeats, var_nfeats

_CONS_NFEATS, _VAR_NFEATS = _compute_feature_dims()

MODEL_PARAMS = {
    'emb_size': 128,
    'cons_nfeats': _CONS_NFEATS,
    'edge_nfeats': 1,
    'var_nfeats': _VAR_NFEATS,
}

# --- Training Hyperparameters ---
TRAIN_PARAMS = {
    'problem': 'facilities',
    'seed': 42,
    'max_epochs': 1000,
    'batch_size': 64,
    'valid_batch_size': 64,
    'lr': 0.0001,
    'patience': 10,
    'early_stopping': 20,
    'num_classes': 50, # the number of disjunctions.
    'temperature': 0.1, # for SupervisedContrastiveLoss
    'num_workers': 8,
}

# --- Test Parameters ---
# TEST_FOLDERS 定义了 `04_test.py` 将要运行评估的文件夹名称列表。
# TEST_CONFIG 存储了每个测试文件夹的特定配置，例如真实的簇数。
# 这里的 'true_n_clusters' 对于CFLP问题来说就是析取的数量，等于'dimension'。
TEST_FOLDERS = ["test", 'artificial']

TEST_CONFIG = {
    'test': {'true_n_clusters': 50},
    'transfer_30_30': {'true_n_clusters': 30},
    'transfer_50_50': {'true_n_clusters': 50},
    'transfer_80_80': {'true_n_clusters': 80},
    'artificial': {'true_n_clusters': 100}
}

# --- Evaluation Parameters ---
EVAL_FOLDERS = ["artificial"]

EVAL_PARAMS = {
    'test_batch_size': 1,
    'dbscan_min_samples': 2,
}
