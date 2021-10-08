#!/bin/bash
#
# Copyright (C) 2018  Sebastian Ruder <sebastian@ruder.io>
# Copyright (C) 2019  Paula Czarnowska <pjc211@cam.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# Contributor(s): 2019 Paula Czarnowska <pjc211@cam.ac.uk>

source activate bli

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DATA="$ROOT/../data"
DICTIONARIES="$DATA/dictionaries"
UNIMORPH="$ROOT/../resources/unimorph"
OUTPUT="$ROOT/../coling"
EMB_DIR="$DATA/embeddings/original"

METHOD_COUNT=3
METHOD_IDS=('artetxe2016' 'ruder2018' 'sgd')
METHOD_TRAIN_ARGS=('--orthogonal' \
    '--lat_var --orthogonal --self_learning --rank_constr 40000 --n_similar 15' \
    '--sgd --optimizer Adam')
METHOD_EVAL_ARGS=('' '--dot' '--dot')
METHOD_EMBEDDINGS=('unit center' 'unit center' 'unit center')

LANGUAGES=('fra' 'spa' 'ita')
LNAMES=('french' 'spanish' 'italian')

# (tuned) parameter values for the sgd translator
declare -A BSIZE=( ['fra-spa']=24 ['fra-ita']=24 ['spa-fra']=24 \
                   ['spa-ita']=24 ['ita-fra']=24 ['ita-spa']=24 )
declare -A REGALPHA=( ['fra-spa']=5 ['fra-ita']=15 ['spa-fra']=15 \
                   ['spa-ita']=10 ['ita-fra']=15 ['ita-spa']=10 )
declare -A LR=( ['fra-spa']=0.025 ['fra-ita']=0.05 ['spa-fra']=0.05 \
                   ['spa-ita']=0.05 ['ita-fra']=0.05 ['ita-spa']=0.05 )

DICTIONARY_COUNT=3
TEST_DICTIONARIES=('morph_dictionaries' 'morph_dictionaries' 'muse')
TRAIN_DICTIONARIES=('morph_dictionaries' 'IDENTICAL' 'muse')
DICTIONARY_TRAIN_ARGS=('' '--identical' '')

export PYTHONPATH=$PYTHONPATH:$ROOT

for ((i = $1; i < $2; i++))
do
    for ((z = $3; z < $4; z++))
    do
        if [[ $i == $z ]]; then
            continue
        fi
        
        sind=$i
        tind=$z
        src=${LANGUAGES[$sind]}
        trg=${LANGUAGES[$tind]}

        echo '--------------------------------------------------------------------------------'
        echo ${LNAMES[$sind]}-${LNAMES[$tind]}
        echo '--------------------------------------------------------------------------------'
        for ((j = 0; j < 1; j++))
        do
            train_dict_name=${TRAIN_DICTIONARIES[$j]}
            test_dict_name=${TEST_DICTIONARIES[$j]}

            for ((k = $5; k < $6; k++))
            do
                model="${METHOD_IDS[$k]}"

                output_dir="$OUTPUT/$src-$trg/$train_dict_name/$model"
                mkdir -p "$output_dir"

                test_dict="$DICTIONARIES/$test_dict_name/$src-$trg.test.txt"
                dev_dict="$DICTIONARIES/$train_dict_name/$src-$trg.dev.txt"
                train_dict="$DICTIONARIES/$train_dict_name/$src-$trg.train.shuf.txt"

                out_matrix="$output_dir/matrix-$src.txt"

                src_vecs="$EMB_DIR/$src-fasttext.emb.txt"
                trg_vecs="$EMB_DIR/$trg-fasttext.emb.txt"
                src_oov="$EMB_DIR/$src-fasttext-oov.emb.txt"
                trg_oov="$EMB_DIR/$trg-fasttext-oov.emb.txt"

                echo "model: $model"
                echo "matrix:" $out_matrix
                if [ ! -f $out_matrix ]; then
                    echo "Learning a mapping..."
                    echo "Dictionary: $train_dict_name"

                    args="${METHOD_TRAIN_ARGS[$k]} ${DICTIONARY_TRAIN_ARGS[$j]} \
                        --normalize ${METHOD_EMBEDDINGS[$k]} \
                        --dictionary $train_dict \
                        --validation $dev_dict \
                        --umsource $UNIMORPH \
                        --precision fp64 \
                        --src_lang $src \
                        --trg_lang $trg"
                    
                    if [ "$model" == "sgd" ]; then
                        extra_args="--lr ${LR[$src-$trg]} --reg_alpha ${REGALPHA[$src-$trg]} \
                            --bs ${BSIZE[$src-$trg]} --model_out $output_dir/model"
                        echo ${extra_args} > $output_dir/"tuned_args.txt"
                    else
                        extra_args=""
                    fi
                    python3 $ROOT/src/train/map_embeddings.py $src_vecs $trg_vecs $out_matrix $args $extra_args -v
                else
                    echo "Mapping already exist."
                fi

                echo $out_matrix
                echo $test_dict_name

                if [ -f $out_matrix ]; then
                    echo "Evaluating the mapping..."
                    echo "matrix: $out_matrix"
                    echo "Dictionary: $test_dict_name"

                    preds_file="$output_dir/predictions-$test_dict_name"
                    dev_test_info="coling-dev-$test_dict_name"
                    test_test_info="coling-test-$test_dict_name"

                    dev_res_file="$output_dir/coling_dev-results-$test_dict_name.txt"
                    test_res_file="$output_dir/coling_test-results-$test_dict_name.txt"

                    oov_args="--src_oov_vecs $src_oov --trg_oov_vecs $trg_oov"

                    eval_args="--original_src_vecs $src_vecs \
                        --original_trg_vecs $trg_vecs \
                        --normalize ${METHOD_EMBEDDINGS[$k]} \
                        --matrix_source $out_matrix \
                        ${METHOD_EVAL_ARGS[$k]} \
                        --precision fp64 \
                        --umsource $UNIMORPH \
                        --src_lang $src --trg_lang $trg \
                        --out_dir $output_dir \
                        $oov_args"

                    # echo -n "  - $model  |  Dev Translation: "
                    # python3 $ROOT/src/test/eval_translation_dev.py ${eval_args} \
                    #     --dictionary $dev_dict \
                    #     --test_info $dev_test_info | tee $dev_res_file # --full_eval

                    echo -n "  - $model  |  Test Translation: "
                    python3 $ROOT/src/test/eval_translation_dev.py ${eval_args} \
                        --dictionary $test_dict \
                        --test_info $test_test_info | tee $test_res_file # --full_eval
                    echo "done $name"
                fi

                if [ -z "$(ls $output_dir)" ]; then
                    rmdir $output_dir
                fi
            done
            echo
        done
        echo
    done
    echo
done
