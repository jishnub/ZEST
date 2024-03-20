from speechbrain.inference.speaker import EncoderClassifier
import os
import torchaudio
import numpy as np
from tqdm import tqdm
from rootpath import DATASET_PATH

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
# classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})
folder = f"{DATASET_PATH}/val"
target_folder = f"{DATASET_PATH}/x_vectors"
os.makedirs(target_folder, exist_ok=True)
wav_files = os.listdir(folder)
wav_files = [x for x in wav_files if ".wav" in x]
wav_files = [x for x in wav_files if ".npy" not in x]

for i, wav_file in enumerate(tqdm(wav_files)):
    sig, sr = torchaudio.load(os.path.join(folder, wav_file))
    embeddings = classifier.encode_batch(sig)[0, 0, :]
    target_file = os.path.join(target_folder, wav_file.replace(".wav", ".npy"))
    np.save(target_file, embeddings.cpu().detach().numpy())
