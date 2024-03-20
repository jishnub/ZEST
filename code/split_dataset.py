import os
import json
import shutil
from rootpath import DATASET_PATH

FullDatasetPath = os.path.join(DATASET_PATH, "Full_Dataset")

def copyfiles(dataset):
    dstdir = os.path.join(DATASET_PATH, dataset)
    with open(f'{dataset}_esd.txt', 'r') as f:
        lines = f.readlines()
        files = [json.loads(s.strip().replace('\'', '\"'))['audio'] for s in lines]
        filenames = sorted([os.path.splitext(os.path.split(f)[-1])[0] for f in files])
        speakers = [t.split('_')[0] for t in filenames]
        for i in set(speakers):
            speakerdir = os.path.join(FullDatasetPath, i)
            detailsfile = os.path.join(speakerdir, f'{i}.txt')
            speaker_filenames = [f for f in filenames if f.startswith(i)]
            with open(detailsfile, 'r') as f:
                lines = [x.strip().split('\t') for x in f.readlines()]
                details = {line[0]:line[2] for line in lines if line[0] in speaker_filenames}

            for f in speaker_filenames:
                src = os.path.join(speakerdir, details[f], f'{f}.wav')
                dst = os.path.join(dstdir, f'{f}.wav')
                shutil.copyfile(src, dst)

if __name__ == "__main__":
    copyfiles("train")
