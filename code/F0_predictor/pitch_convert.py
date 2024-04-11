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
# import random
# import torch.nn.functional as F
from config import hparams, f0_predictor_path, OUTDIR, DATASET_PATH
# import pickle
# import ast
from pitch_attention_adv import create_dataset, PitchModel
from pathlib import Path

pred_DSDT_f0_folder = OUTDIR/"pred_DSDT_f0"

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

# smaller list for testing
sources = ["0011_000021.wav", "0012_000022.wav"]
# sources = ["0011_000021.wav", "0012_000022.wav", "0013_000025.wav",
#            "0014_000032.wav", "0015_000034.wav", "0016_000035.wav",
#            "0017_000038.wav", "0018_000043.wav", "0019_000023.wav",
#            "0020_000047.wav"]

def get_f0(sources = sources, dataset = "test",
            dataset_path = DATASET_PATH, pred_DSDT_f0_folder = pred_DSDT_f0_folder):

    os.makedirs(pred_DSDT_f0_folder, exist_ok=True)
    target_loader = create_dataset(dataset, 1, dataset_path)
    model = PitchModel(hparams)
    model.load_state_dict(torch.load(f0_predictor_path, map_location=device))
    model.to(device)
    model.eval()
    source_loader = create_dataset(dataset, 1, dataset_path, sources)

    with torch.no_grad():
        for source in tqdm(sources):
            source_name = Path(source).stem
            srcspeakerid = source[:5]

            for data in source_loader:
                if not data["names"][0] == source:
                    continue

                speaker_s, mask_s, tokens_s = torch.tensor(data["speaker"]).to(device),\
                                                    torch.tensor(data["mask"]).to(device),\
                                                    torch.tensor(data["hubert"]).to(device)


            for data in target_loader:
                names = data["names"]
                inputs_t, labels_t = torch.tensor(data["audio"]).to(device),\
                                        torch.tensor(data["labels"]).to(device)

                if ((not names[0].startswith(srcspeakerid)) and labels_t[0] > 0 and
                        (int(names[0][5:11]) - int(source[5:11]))%350 != 0):

                    pitch_pred, _, _, _ = model(inputs_t, tokens_s, speaker_s, mask_s)
                    pitch_pred = torch.exp(pitch_pred) - 1
                    final_name = source_name + names[0]
                    final_name = Path(final_name).with_suffix(".npy")
                    np.save(pred_DSDT_f0_folder/final_name, pitch_pred[0, :].cpu().detach().numpy())

if __name__ == "__main__":
    get_f0()
