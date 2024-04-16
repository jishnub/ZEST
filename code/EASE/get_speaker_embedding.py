from speechbrain.inference.speaker import EncoderClassifier
import os
import torchaudio
import numpy as np
from tqdm import tqdm
from pathlib import Path

HOMEDIR = Path.home()
OUTDIR = HOMEDIR/"ZEST_data"
DATASET_PATH = HOMEDIR/"Emotional_Speech_Dataset"
XVECTORS_FOLDER = OUTDIR/"x_vectors"

# classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})
def getembeddings(dataset_path = DATASET_PATH):
    wav_folder = Path(dataset_path)
    os.makedirs(XVECTORS_FOLDER, exist_ok=True)
    wav_files = [Path(x) for x in os.listdir(wav_folder) if Path(x).suffix == ".wav"]

    for wav_file in tqdm(wav_files):
        sig, _ = torchaudio.load(wav_folder/wav_file)
        embeddings = classifier.encode_batch(sig)[0, 0, :]
        target_file = XVECTORS_FOLDER/wav_file.with_suffix(".npy")
        np.save(target_file, embeddings.cpu().detach().numpy())

if __name__ == '__main__':
    for label in ["train", "val", "test"]:
        dataset_path = DATASET_PATH/label
        getembeddings(dataset_path)
