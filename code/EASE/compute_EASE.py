import get_speaker_embedding
import speaker_classifier
from pathlib import Path

HOMEDIR = Path.home()
datasets = "unseenaudio"
datasets_path = HOMEDIR

def compute_ease(datasets=datasets, datasets_path=datasets_path):
    get_speaker_embedding.getembeddings(dataset_path=datasets_path/datasets)
    speaker_classifier.get_embedding(datasets=[datasets], dataset_path=datasets_path)

if __name__ == "__main__":
    compute_ease()
