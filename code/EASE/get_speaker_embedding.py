from speechbrain.inference.speaker import EncoderClassifier
import os
import torchaudio
import numpy as np
from tqdm import tqdm
from pathlib import Path
home = Path.home()
OUTDIR = home/"ZEST_data"
DATASET_PATH = f"{home}/Emotional_Speech_Dataset"

# classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})
def getembeddings(folder):
    folder = DATASET_PATH/folder
    target_folder = OUTDIR/"x_vectors"
    os.makedirs(target_folder, exist_ok=True)
    wav_files = [Path(x) for x in os.listdir(folder) if Path(x).suffix == ".wav"]

    for wav_file in tqdm(wav_files):
        sig, _ = torchaudio.load(folder/wav_file)
        embeddings = classifier.encode_batch(sig)[0, 0, :]
        target_file = target_folder/wav_file.with_suffix(".npy")
        np.save(target_file, embeddings.cpu().detach().numpy())

if __name__ == '__main__':
    getembeddings("train")
    getembeddings("val")
    getembeddings("test")
