import os
import torch # type: ignore
import logging
import numpy as np
import random
from tqdm import tqdm
from config import hparams, f0_predictor_path, OUTDIR, DATASET_PATH
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
sources = ["0011_000021.wav"]
# sources = ["0011_000021.wav", "0012_000022.wav", "0013_000025.wav",
#            "0014_000032.wav", "0015_000034.wav", "0016_000035.wav",
#            "0017_000038.wav", "0018_000043.wav", "0019_000023.wav",
#            "0020_000047.wav"]

def custom_collate(data):
    # batch_len = len(data)
    new_data = {"audio":[], "mask":[], "labels":[], "hubert":[], "speaker":[], "names":[]}
    max_len_hubert, max_len_aud = 0, 0
    for ind in range(len(data)):
        max_len_aud = max(data[ind][0].shape[-1], max_len_aud)
        max_len_hubert = max(data[ind][2].shape[-1], max_len_hubert)
    for i in range(len(data)):
        final_sig = np.concatenate((data[i][0], np.zeros((max_len_aud-data[i][0].shape[-1]))), -1)
        mask = data[i][2].shape[-1]
        hubert_feat = np.concatenate((data[i][2], 100*np.ones((max_len_hubert-data[i][2].shape[-1]))), -1)
        labels = data[i][3]
        speaker_feat = data[i][4]
        names = data[i][6]
        new_data["audio"].append(final_sig)
        new_data["mask"].append(torch.tensor(mask))
        new_data["hubert"].append(hubert_feat)
        new_data["labels"].append(torch.tensor(labels))
        new_data["speaker"].append(speaker_feat)
        new_data["names"].append(names)
    new_data["audio"] = np.array(new_data["audio"])
    new_data["mask"] = np.array(new_data["mask"])
    new_data["hubert"] = np.array(new_data["hubert"])
    new_data["labels"] = np.array(new_data["labels"])
    new_data["speaker"] = np.array(new_data["speaker"])
    return new_data

def get_f0(sources = sources, dataset = "test",
            dataset_path = DATASET_PATH, pred_DSDT_f0_folder = pred_DSDT_f0_folder):

    os.makedirs(pred_DSDT_f0_folder, exist_ok=True)
    model = PitchModel(hparams)
    model.load_state_dict(torch.load(f0_predictor_path, map_location=device))
    model.to(device)
    model.eval()
    target_loader = create_dataset(dataset, 1, dataset_path, collate_fn=custom_collate)
    source_loader = create_dataset(dataset, 1, dataset_path, sources, collate_fn=custom_collate)

    with torch.no_grad():
        for source in tqdm(sources):
            source_name = Path(source).stem

            for data in source_loader:
                if not data["names"][0] == source:
                    continue

                speaker_s, mask_s, tokens_s = torch.tensor(data["speaker"]).to(device),\
                                                    torch.tensor(data["mask"]).to(device),\
                                                    torch.tensor(data["hubert"]).to(device)

            for data in target_loader:
                names = data["names"]
                inputs_t = torch.tensor(data["audio"]).to(device)

                pitch_pred, _, _, _ = model(inputs_t, tokens_s, speaker_s, mask_s)
                pitch_pred = torch.exp(pitch_pred) - 1
                final_name = source_name + names[0]
                final_name = Path(final_name).with_suffix(".npy")
                final_path = pred_DSDT_f0_folder/final_name
                np.save(final_path, pitch_pred[0, :].cpu().detach().numpy())

if __name__ == "__main__":
    get_f0()
