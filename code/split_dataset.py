import os
import json
import shutil
from rootpath import DATASET_PATH

FullDatasetPath = os.path.join(DATASET_PATH, "Full_Dataset")

def string_to_dict(s):
    return json.loads(s.strip().replace('\'', '\"'))

def filename_from_line(s):
    f = string_to_dict(s)['audio']
    filename = os.path.splitext(os.path.split(f)[-1])[0]
    return filename

def speaker_name_from_filename(t):
    return t.split('_')[0]

def read_esd_file(dataset):
    with open(f'{dataset}_esd.txt', 'r') as f:
        lines = sorted(f.readlines(), key=filename_from_line)
        filenames = [filename_from_line(s) for s in lines]
        speakers = [speaker_name_from_filename(t) for t in filenames]

    return set(speakers), filenames, lines

def replace_file_path(s, dstdir):
    d = string_to_dict(s)
    f = d["audio"]
    d["audio"] = os.path.join(dstdir, os.path.split(f)[-1])
    return json.dumps(d)

def copyfiles(dataset, nfiles_per_speaker=10_000):
    dstdir = os.path.join(DATASET_PATH, dataset)
    speakers, filenames, _ = read_esd_file(dataset)
    for speakerno in speakers:
        speakerdir = os.path.join(FullDatasetPath, speakerno)
        detailsfile = os.path.join(speakerdir, f'{speakerno}.txt')
        speaker_filenames = [f for f in filenames if f.startswith(speakerno)][:nfiles_per_speaker]
        with open(detailsfile, 'r') as f:
            lines = [x.strip().split('\t') for x in f.readlines()]
            details = {line[0]:line[2] for line in lines if line[0] in speaker_filenames}

        for f in speaker_filenames:
            src = os.path.join(speakerdir, details[f], f'{f}.wav')
            dst = os.path.join(dstdir, f'{f}.wav')
            if not os.path.isfile(dst):
                shutil.copyfile(src, dst)

def trim_esd_file(dataset, nfiles_per_speaker=10_000):
    dstdir = os.path.join(DATASET_PATH, dataset)
    speakers, _, lines = read_esd_file(dataset)
    lines_trimmed = []
    speaker_count = {s:0 for s in speakers}
    for s in lines:
        filename = filename_from_line(s)
        speaker_name = speaker_name_from_filename(filename)
        if speaker_count[speaker_name] < nfiles_per_speaker:
            s = replace_file_path(s.strip(), dstdir)
            lines_trimmed.append(s.strip())
            speaker_count[speaker_name] += 1
        # check if done
        if all(val == nfiles_per_speaker for val in speaker_count.values()):
            break

    with open(f'{dataset}_esd_trimmed.txt', 'w') as f:
        f.write('\n'.join(lines_trimmed))

