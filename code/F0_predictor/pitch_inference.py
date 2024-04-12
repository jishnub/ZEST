import os
import torch # type: ignore
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

def collatefn_f0_inference(data):
    new_data = {"audio":[], "mask":[], "hubert":[], "speaker":[], "names":[]}
    max_len_hubert, max_len_aud = 0, 0
    for ind in range(len(data)):
        max_len_aud = max(data[ind][0].shape[-1], max_len_aud)
        max_len_hubert = max(data[ind][2].shape[-1], max_len_hubert)
    for i in range(len(data)):
        final_sig = np.concatenate((data[i][0], np.zeros((max_len_aud-data[i][0].shape[-1]))), -1)
        mask = data[i][2].shape[-1]
        hubert_feat = np.concatenate((data[i][2], 100*np.ones((max_len_hubert-data[i][2].shape[-1]))), -1)
        speaker_feat = data[i][4]
        names = data[i][6]

        new_data["speaker"].append(speaker_feat)
        new_data["audio"].append(final_sig)
        new_data["mask"].append(torch.tensor(mask))
        new_data["hubert"].append(hubert_feat)
        new_data["names"].append(names)

    new_data["speaker"] = np.array(new_data["speaker"])
    new_data["audio"] = np.array(new_data["audio"])
    new_data["mask"] = np.array(new_data["mask"])
    new_data["hubert"] = np.array(new_data["hubert"])
    return new_data

def predict_pitch_contours(loader, model):
    for data in tqdm(loader):
            names = data["names"]
            speaker, inputs, mask, tokens = torch.tensor(data["speaker"]).to(device),\
                                    torch.tensor(data["audio"]).to(device),\
                                    torch.tensor(data["mask"]).to(device),\
                                    torch.tensor(data["hubert"]).to(device)

            pitch_pred, _, _, _ = model(inputs, tokens, speaker, mask)
            pitch_pred = torch.exp(pitch_pred) - 1
            for ind,filename in enumerate(names):
                target_file_name = Path(filename).with_suffix(".npy").name
                np.save(CONTOURDIR/target_file_name, pitch_pred[ind, :].cpu().detach().numpy())

def get_f0(datasets = ["test", "train", "val"], dataset_path = DATASET_PATH):
    os.makedirs(CONTOURDIR, exist_ok=True)

    model = PitchModel(hparams)
    model.load_state_dict(torch.load(f0_predictor_path, map_location=device))
    model.to(device)
    model.eval()
    loaders = [create_dataset(label, bs=1,
                    dataset_path=dataset_path,
                    collate_fn=collatefn_f0_inference) for label in datasets]
    with torch.no_grad():
        for loader in loaders:
            predict_pitch_contours(loader, model)


if __name__ == "__main__":
    get_f0()
