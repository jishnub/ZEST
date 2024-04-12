import os
import torch
import logging
import numpy as np
import random
from tqdm import tqdm
from config import hparams, DATASET_PATH
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

def custom_collate(data):
    # batch_len = len(data)
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

class HiddenEmoPitchModel(PitchModel):
    def forward(self, aud, tokens=None, speaker=None, lengths=None, alpha=1.0):
        inputs = self.processor(aud, sampling_rate=16000, return_tensors="pt")
        _, _, emo_hidden, _ = self.encoder(inputs['input_values'].to(device), alpha)

        return emo_hidden

def compute_wav2vec_feats(loader, model):
    for data in tqdm(loader):
            name = data["names"]
            speaker, inputs, mask ,tokens = torch.tensor(data["speaker"]).to(device),\
                                            torch.tensor(data["audio"]).to(device), \
                                            torch.tensor(data["mask"]).to(device),\
                                            torch.tensor(data["hubert"]).to(device)

            embedded = model(inputs, tokens, speaker, mask)
            for ind in range(len(name)):
                target_file_name = Path(name[ind]).with_suffix(".npy")
                np.save(wav2vec_feats_folder/target_file_name, embedded[ind, :].cpu().detach().numpy())

def train(datasets = ["train", "test", "val"], dataset_path = DATASET_PATH):
    os.makedirs(wav2vec_feats_folder, exist_ok=True)
    loaders = [create_dataset(label, 1, dataset_path, collate_fn=custom_collate) for label in datasets]
    model = HiddenEmoPitchModel(hparams)
    model.load_state_dict(torch.load(f0_predictor_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for loader in loaders:
            compute_wav2vec_feats(loader, model)

if __name__ == "__main__":
    train()
