import os
from pathlib import Path

# Paths of project
PROJECT_PATH = Path(__file__).parent.parent
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
PATCHES_PATH = os.path.join(DATA_PATH, 'patches')
TRAIN_DATA_PATH = os.path.join(PATCHES_PATH, 'train')
TEST_DATA_PATH = os.path.join(PATCHES_PATH, 'test')
VAL_DATA_PATH = os.path.join(PATCHES_PATH, 'val')
MODELS_PATH = os.path.join(PROJECT_PATH, 'models')

# Object detection training configuration
CLASSES = ['__background__', 'house']
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 0.005

# Object detection inference configuration
SCORE_THRESHOLD = 0.5
