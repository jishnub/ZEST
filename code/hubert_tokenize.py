import wave
import os

FILEDIR = os.path.dirname(os.path.realpath(__file__))

# this manifest file is an input to fairseq
# the number of frames are read in from the actual audio files
def generate_manifest(audiopath, manifestoutputdir = FILEDIR):
    files = [f for f in os.listdir(audiopath) if os.path.splitext(f)[-1] == ".wav"]
    files.sort()
    manifest_path = os.path.join(manifestoutputdir, "Manifest")
    with open(manifest_path, "w") as manifest:
        manifest.write(audiopath + "\n")
        for wavfile in files:
            wavfilepath = os.path.join(audiopath, wavfile)
            with wave.open(wavfilepath, "r") as wavf:
                nf = wavf.getnframes()
            manifest.write(wavfile + "\t" + str(nf) + "\n")

# convert the output quantized file to the ZEST format used in the code
# audio durations are read in from the actual audio files
def fairseq_tokenfile_to_zest_format(inputfile, outputfile, audiopath):
    with open(inputfile, "r") as inpf, open(outputfile, "w") as outf:
        for line in inpf:
            wavfile, tokens = line.rstrip("\n").split("|")
            wavpath = os.path.join(audiopath, wavfile + ".wav")
            with wave.open(wavpath, "r") as wavf:
                duration = wavf.getnframes() / wavf.getframerate()
            outstr = f'{{"audio": "{wavpath}", "hubert": "{tokens}", "duration": {duration}}}'
            outf.write(outstr + "\n")
