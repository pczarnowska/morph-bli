#!/bin/bash
arch=$1
pair=$2
datsource=$3
testsource=$4

root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
outdir="$root/../model_reinfl_$datsource/monotag-$arch/$pair"

if [[ -d $outdir ]]; then
    echo ""
    echo "$outdir exists, only evaluating..."
    echo ""
    model_file=$(ls ${outdir}/$pair.nll* | head -1)
    python $root/../src/train.py \
        --dataset sigmorphon19task1 \
        --train $root/../$datsource/$pair.tag.train.txt  \
        --dev $root/../$testsource/$pair.tag.dev.txt \
        --test $root/../$testsource/$pair.tag.test.txt \
        --model $outdir/$pair --seed 0 \
        --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
        --src_layer 2 --trg_layer 1 --max_norm 5 \
        --arch $arch --estop 1e-8 --epochs 0 --bs 20 --mono \
        --load ${model_file} 
else
    echo ""
    echo "$outdir doesn't exist, training..."
    echo ""
    python $root/../src/train.py \
        --dataset sigmorphon19task1 \
        --train $root/../$datsource/$pair.tag.train.txt  \
        --dev $root/../$datsource/$pair.tag.dev.txt \
        --model $outdir/$pair --seed 0 \
        --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
        --src_layer 2 --trg_layer 1 --max_norm 5 \
        --arch $arch --estop 1e-8 --epochs 50 --bs 20 --mono
fi
