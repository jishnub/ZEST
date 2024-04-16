import get_wav2vec_feats
import pitch_inference
import pitch_convert
from pathlib import Path
import os

HOMEDIR = Path.home()
datasets = ["unseen"]
dataset_path = HOMEDIR/"unseenaudio"

sources = [f for f in os.listdir(dataset_path) if Path(f).suffix == ".wav"]

def infer_convert(datasets=datasets, dataset_path=dataset_path, sources = sources):
    pitch_inference.get_f0(datasets=datasets, dataset_path=dataset_path)
    get_wav2vec_feats.train(datasets=datasets, dataset_path=dataset_path)
    pitch_convert.get_f0(sources=sources, dataset=datasets[0], dataset_path=dataset_path)

if __name__ == "__main__":
    infer_convert()
