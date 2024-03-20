import os
from pathlib import Path
home = str(Path.home())
CODE_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_PATH = f"{home}/Emotional_Speech_Dataset"
