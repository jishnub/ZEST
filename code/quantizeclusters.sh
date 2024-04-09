#!/bin/bash
set -e

# Generate HuBERT tokens using fairseq. This generates the train_esd.txt, test_esd.txt and val_esd.txt files.
# Download and install fairseq from https://github.com/facebookresearch/fairseq
# the code below is adapted from https://github.com/facebookresearch/fairseq/blob/bedb259bf34a9fc22073c13a1cee23192fa70ef3/examples/textless_nlp/gslm/speech2unit/README.md
# Copyright (c) Facebook, Inc. and its affiliates.
# fairseq is released under the following license:
# MIT License
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# Adjust the paths accordingly

LAYER=6
N_CLUSTERS=100
TYPE="hubert"

# Download the hubert base model from https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
HUBERTMODELDIR=$HOME/fairseq/hubert_trained_model
CKPT_PATH=$HUBERTMODELDIR/hubert_base_ls960.pt # update this path
# Download the pretrained quantized model from https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin
KM_MODEL_PATH=$HUBERTMODELDIR/km.bin  # update this path

ZESTDIR=$HOME/ZEST
ZESTCODEDIR=$ZESTDIR/code
MANIFEST_OUTPUT_DIR=$ZESTCODEDIR # update this path
MANIFEST=$MANIFEST_OUTPUT_DIR/Manifest
OUT_QUANTIZED_FILE=$MANIFEST_OUTPUT_DIR/quantized.txt # output file

AUDIODIR="$HOME/Emotional_Speech_Dataset" # update to the directory containing wav files

source $ZESTDIR/.venv/bin/activate
PYTHONPATH=$ZESTCODEDIR python -c "import hubert_tokenize as htoken; htoken.generate_manifest('$AUDIODIR', '$MANIFEST_OUTPUT_DIR')"
deactivate

FAIRSEQDIR=$HOME/fairseq
source $FAIRSEQDIR/.venv/bin/activate
PYTHONPATH=$FAIRSEQDIR python $FAIRSEQDIR/examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension ".wav"

deactivate

ESDFILE=$MANIFEST_OUTPUT_DIR/test_esd.txt
source $ZESTDIR/.venv/bin/activate
PYTHONPATH=$ZESTCODEDIR python -c "import hubert_tokenize as htoken; htoken.fairseq_tokenfile_to_zest_format('$OUT_QUANTIZED_FILE', '$ESDFILE', '$AUDIODIR')"
deactivate
