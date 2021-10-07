#!/usr/bin/env bash

source activate bli

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DATA="$ROOT/../data/dictionaries/morph_dictionaries"
OUT=$1
OUT_WITH_UM=$2
UM_SOURCE="$ROOT/../resources/unimorph"

LANGS=( "fra" "spa" "ita" )


for ((i = 0; i < ${#LANGS[@]}; i++))
do
    for ((j = 0; j < ${#LANGS[@]}; j++))
        do
            if [[ $i == $j ]]; then
                continue
            fi

            l1=${LANGS[$i]}
            l2=${LANGS[$j]}

            echo $l1 $l2

            # go_beyond_um adds other POS like adverbs, pronouns etc.
            python3 src/split_unimorph.py --lang1 $l1 --lang2 $l2 --base_dict_dir $DATA --out_dir $OUT
            python3 src/split_unimorph.py --lang1 $l1 --lang2 $l2 --base_dict_dir $DATA --out_dir $OUT_WITH_UM --include_unimorph --umsource $UM_SOURCE # --go_beyond_unimorph

        done
done
