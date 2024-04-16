import os
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn as nn
import random
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.autograd import Function
from pathlib import Path
from get_speaker_embedding import XVECTORS_FOLDER, DATASET_PATH, OUTDIR

EMBEDDINGDIR = OUTDIR/"EASE_embeddings"

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

class MyDataset(Dataset):
    def __init__(self, wav_folder, xvector_folder, wav_files = None):
        if wav_files is None:
            wav_files = [x for x in os.listdir(wav_folder) if Path(x).suffix == ".wav"]

        speaker_features = [Path(x).with_suffix(".npy") for x in wav_files
                                if (Path(xvector_folder)/Path(x).with_suffix(".npy")).is_file()]

        if not len(wav_files) == len(speaker_features):
            raise Exception(f"number of wav files ({len(wav_files)}) doesn't match "+
                            f"the number of x-vectors: ({len(speaker_features)})")

        self.folder = Path(wav_folder)
        self.speaker_folder = Path(xvector_folder)
        self.wav_files = wav_files
        self.speaker_features = speaker_features
        self.sr = 16000

    def __len__(self):
        return len(self.wav_files)

    def getspkrlabel(self, file_name):
        spkr_name = file_name.split("_")[0]
        spkr_label = int(spkr_name) - 11

        return spkr_label

    def getemolabel(self, file_name):
        file_name = int(Path(file_name).stem.split("_")[-1])
        return (file_name-1) // 350

    def __getitem__(self, audio_ind):
        speaker_feat = np.load(self.speaker_folder/self.speaker_features[audio_ind])
        speaker_label = self.getspkrlabel(self.wav_files[audio_ind])
        class_id = self.getemolabel(self.wav_files[audio_ind])

        return speaker_feat, speaker_label, class_id, self.wav_files[audio_ind]

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class SpeakerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(192, 128)
        self.fc = nn.Linear(128, 128)
        self.fc_embed = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_embed_1 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self.fc4 = nn.Linear(128, 128)
        self.fc_embed_2 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 5)

    def forward(self, feat, alpha=1.0):
        feat = self.fc(self.fc_embed(self.fc1(feat)))
        reverse = ReverseLayerF.apply(feat, alpha)
        out = self.fc3(self.fc_embed_1(self.fc2(feat)))
        emo_out = self.fc5(self.fc_embed_2(self.fc4(reverse)))

        return out, emo_out, feat

def create_dataset(mode, bs=32, dataset_path = DATASET_PATH):
    wav_folder = Path(dataset_path)/mode
    dataset = MyDataset(wav_folder, XVECTORS_FOLDER)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False)

    return loader

def train():

    train_loader = create_dataset("train")
    val_loader = create_dataset("val")
    model = SpeakerModel()
    model.to(device)
    base_lr = 1e-4
    parameters = list(model.parameters())
    optimizer = Adam([{'params':parameters, 'lr':base_lr}])
    final_val_loss = 1e20

    for e in range(10):
        model.train()
        tot_loss = 0.0
        val_loss = 0.0
        pred_tr = []
        gt_tr = []
        pred_val = []
        gt_val = []
        for i, data in enumerate(tqdm(train_loader)):
            model.train()
            p = float(i + e * len(train_loader)) / 100 / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            speaker_feat, labels, emo_labels = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs, out_emo, _ = model(speaker_feat, alpha)
            loss = nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
            loss_emo = nn.CrossEntropyLoss(reduction='mean')(out_emo, emo_labels)
            loss += 10*loss_emo
            tot_loss += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.argmax(outputs, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_tr.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_tr.extend(labels)

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                speaker_feat, labels, emo_labels = data[0].to(device), data[1].to(device), data[2].to(device)
                outputs, out_emo, _ = model(speaker_feat)
                loss = nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
                val_loss += loss.detach().item()
                pred = torch.argmax(outputs, dim = 1)
                pred = pred.detach().cpu().numpy()
                pred = list(pred)
                pred_val.extend(pred)
                labels = labels.detach().cpu().numpy()
                labels = list(labels)
                gt_val.extend(labels)
        if val_loss < final_val_loss:
            torch.save(model.state_dict(), 'EASE.pth')
            final_val_loss = val_loss
        train_loss = tot_loss/len(train_loader)
        train_f1 = accuracy_score(gt_tr, pred_tr)
        val_loss_log = val_loss/len(val_loader)
        val_f1 = accuracy_score(gt_val, pred_val)
        e_log = e + 1
        logger.info(f"Epoch {e_log}, \
                    Training Loss {train_loss},\
                    Training Accuracy {train_f1}")
        logger.info(f"Epoch {e_log}, \
                    Validation Loss {val_loss_log},\
                    Validation Accuracy {val_f1}")


def compute_and_save_embedding(loader, model):
    for data in tqdm(loader):
        speaker_feat = data[0].to(device)
        names = data[3]
        _, _, embedded = model(speaker_feat)
        for ind in range(len(names)):
            target_file_name = Path(names[ind]).with_suffix(".npy")
            np.save(EMBEDDINGDIR/target_file_name, embedded[ind, :].cpu().detach().numpy())


def get_embedding(datasets = ["train", "val", "test"], dataset_path = DATASET_PATH):
    loaders = [create_dataset(label, 1, dataset_path) for label in datasets]
    model = SpeakerModel()
    model = torch.load_state_dict('EASE.pth', map_location=device)
    model.to(device)
    model.eval()
    os.makedirs(EMBEDDINGDIR, exist_ok=True)
    with torch.no_grad():
        for loader in loaders:
            compute_and_save_embedding(loader, model)

if __name__ == "__main__":
    train()
    get_embedding()
