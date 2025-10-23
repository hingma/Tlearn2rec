import os
from pathlib import Path

# Base paths (scoped to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Default dataset/problem
PROBLEM = 'osif'

# Data splits: directories already exist under data/processed/<problem>/{train,valid,test}
TRAIN_DIR = PROCESSED_DIR / PROBLEM / 'train'
VALID_DIR = PROCESSED_DIR / PROBLEM / 'valid'
TEST_DIR = PROCESSED_DIR / PROBLEM / 'test'

# Training hyperparameters
SEED = 42
BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
LR = 1e-3
MAX_EPOCHS = 200
PATIENCE = 10
EARLY_STOPPING = 20
TEMPERATURE = 0.1
NUM_WORKERS = min(8, os.cpu_count() or 2)

# Model hyperparameters (encoder dims)
EMBED_DIM = 64
HIDDEN_DIM = 64

# Output experiment directory
EXPERIMENT_DIR = MODELS_DIR / PROBLEM / 'UnsupervisedGNN'
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


# Debug/printing controls
DEBUG_SHAPES = True  # print tensor and layer shapes during the first steps


