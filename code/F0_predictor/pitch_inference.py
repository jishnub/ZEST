import os
import torch
# import torchaudio
# from einops.layers.torch import Rearrange
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import logging
import numpy as np
# import json
# from torch.utils.data import DataLoader, Dataset
# from torch.optim import Adam
# import torch.nn as nn
import random
# from sklearn.metrics import f1_score
from tqdm import tqdm
# import torch.nn.functional as F
from config import hparams, OUTDIR, f0_predictor_path, DATASET_PATH
# import pickle
# import ast
from pitch_attention_adv import create_dataset, PitchModel
# from torch.autograd import Function
from pathlib import Path

CONTOURDIR = OUTDIR/"f0_contours"

torch.set_printoptions(profile="full")
#Logger set
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
torch.autograd.set_detect_anomaly(True)
#CUDA devices enabled
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

def predict_pitch_contours(loader, model):
    for data in tqdm(loader):
            inputs, mask ,tokens = torch.tensor(data["audio"]).to(device), \
                                                   torch.tensor(data["mask"]).to(device),\
                                                   torch.tensor(data["hubert"]).to(device)

            speaker = torch.tensor(data["speaker"]).to(device)
            pitch_pred, _, _, _ = model(inputs, tokens, speaker, mask)
            pitch_pred = torch.exp(pitch_pred) - 1
            names = data["names"]
            for ind,filename in enumerate(names):
                target_file_name = Path(filename).with_suffix(".npy").name
                np.save(CONTOURDIR/target_file_name, pitch_pred[ind, :].cpu().detach().numpy())

def get_f0(datasets = ["test", "train", "val"], dataset_path = DATASET_PATH):
    os.makedirs(CONTOURDIR, exist_ok=True)

    model = PitchModel(hparams)
    model.load_state_dict(torch.load(f0_predictor_path, map_location=device))
    model.to(device)
    model.eval()
    loaders = [create_dataset(label, 1, dataset_path) for label in datasets]
    with torch.no_grad():
        for loader in loaders:
            predict_pitch_contours(loader, model)


if __name__ == "__main__":
    get_f0()
