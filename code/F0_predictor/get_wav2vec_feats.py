import os
import torch
# from einops.layers.torch import Rearrange
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import logging
import numpy as np
# from torch.utils.data import DataLoader, Dataset
# from torch.optim import Adam
# import torch.nn as nn
import random
# from sklearn.metrics import f1_score
from tqdm import tqdm
# import random
# import torch.nn.functional as F
from config import hparams, DATASET_PATH
# from config import f0_file
# import ast
# import math
# from torch.autograd import Function
# from pitch_convert import crema_dataset
from pitch_attention_adv import create_dataset, PitchModel
from pathlib import Path
from config import f0_predictor_path, OUTDIR

wav2vec_feats_folder = OUTDIR/"wav2vec_feats"

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

class HiddenEmoPitchModel(PitchModel):
    def forward(self, aud, tokens=None, speaker=None, lengths=None, alpha=1.0):
        inputs = self.processor(aud, sampling_rate=16000, return_tensors="pt")
        _, _, emo_hidden, _ = self.encoder(inputs['input_values'].to(device), alpha)

        return emo_hidden

def compute_wav2vec_feats(loader, model):
    for data in tqdm(loader):
            inputs, mask ,tokens = torch.tensor(data["audio"]).to(device), \
                                                   torch.tensor(data["mask"]).to(device),\
                                                   torch.tensor(data["hubert"]).to(device)

            speaker = torch.tensor(data["speaker"]).to(device)
            name = data["names"]
            embedded = model(inputs, tokens, speaker, mask)
            for ind in range(len(name)):
                target_file_name = Path(name[ind]).with_suffix(".npy")
                np.save(wav2vec_feats_folder/target_file_name, embedded[ind, :].cpu().detach().numpy())

def train(datasets = ["train", "test", "val"], dataset_path = DATASET_PATH):
    os.makedirs(wav2vec_feats_folder, exist_ok=True)
    loaders = [create_dataset(label, 1, dataset_path) for label in datasets]
    model = HiddenEmoPitchModel(hparams)
    model.load_state_dict(torch.load(f0_predictor_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for loader in loaders:
            compute_wav2vec_feats(loader, model)

if __name__ == "__main__":
    train()
