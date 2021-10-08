
#!/bin/bash
#
# Copyright (C) 2018  Sebastian Ruder <sebastian@ruder.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#
# Contributor(s): 2019 Paula Czarnowska <pjc211@cam.ac.uk>

source activate bli

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REINFLROOT="$ROOT/../reinflector"

DATA="$ROOT/../data"
DICTIONARIES="$DATA/dictionaries"
UNIMORPH="$ROOT/../resources/unimorph"
OUTPUT="$ROOT/../coling"
EMB_DIR="$DATA/embeddings/original"

RSET_COUNT=1
REINFL_MODELS=("model_reinfl_reinfl_data_um")  # "model_reinfl_reinfl_data_um_beyond_um" (if muse)
ANALYSIS_MODELS=("model_analyze_analyze_data_um") # "model_analyze_analyze_data_beyond_um" (if muse)
OUTPUT="$ROOT/../coling"

LANGUAGES=('fra' 'spa' 'ita')
LNAMES=('french' 'spanish' 'italian')

DICTIONARY_COUNT=1
TEST_DICTIONARIES=('morph_dictionaries')  # 'muse')
TRAIN_DICTIONARIES=('morph_dictionaries')  # 'muse')


export PYTHONPATH=$PYTHONPATH:$ROOT

MYID=$((1 + RANDOM % 1000))
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

        for ((j = $5; j < $6; j++))
        do
            train_dict_name=${TRAIN_DICTIONARIES[$j]}
            test_dict_name=${TEST_DICTIONARIES[$j]}

            # model dir
            output_dir="$OUTPUT/$src-$trg/$train_dict_name/sgd"
            echo "Evaluating model in $output_dir"

            test_dict="$DICTIONARIES/$test_dict_name/$src-$trg.test.txt"
            dev_dict="$DICTIONARIES/$train_dict_name/$src-$trg.dev.txt"
            train_dict="$DICTIONARIES/$train_dict_name/$src-$trg.train.shuf.txt"

            out_matrix="$output_dir/matrix-$src.txt"
            out_model="$output_dir/model"

            src_um=$UNIMORPH/$src/$src
            trg_um=$UNIMORPH/$trg/$trg

            src_vecs="$EMB_DIR/$src-fasttext.emb.txt"
            trg_vecs="$EMB_DIR/$trg-fasttext.emb.txt"
            src_oov="$EMB_DIR/$src-fasttext-oov.emb.txt"
            trg_oov="$EMB_DIR/$trg-fasttext-oov.emb.txt"

            if [ ! -f $out_matrix ]; then
                echo "Mapping doesn't exist: $out_matrix"
            else
                echo "Evaluating the mapping..."
                echo "matrix: $out_matrix"
                echo "Dictionary: $test_dict_name"

                for ((m = 0; m < $RSET_COUNT; m++))
                do
                    rm_name=${REINFL_MODELS[$m]}
                    am_name=${ANALYSIS_MODELS[$m]}

                    # echo "reinfl. model: " $rm_name
                    # echo "analysis model:" $am_name

                    rm_source="$REINFLROOT/${rm_name}/monotag-hmmfull"
                    am_source="$REINFLROOT/${am_name}/monotag-hmmfull"

                    if [[ $7 == 0 ]]; then
                        key="$test_dict_name-$rm_name-$am_name"
                    elif [[ $7 == 1 ]]; then
                        key="$test_dict_name-hybrid-$rm_name-$am_name"
                    elif [[ $7 == 2 ]]; then
                        key="$test_dict_name-$rm_name-noana"
                    fi

                    echo $key
                    preds_file="$output_dir/predictions-$key.txt"
                    dev_test_info="coling-dev-$key"
                    test_test_info="coling-test-$key"

                    dev_res_file="$output_dir/dev-results-$key.txt"
                    test_res_file="$output_dir/test-results-$key.txt"

                    oov_args="--src_oov_vecs $src_oov --trg_oov_vecs $trg_oov"


                    eval_args="--original_src_vecs $src_vecs \
                        --original_trg_vecs $trg_vecs \
                        --normalize unit center \
                        --dot \
                        --matrix_source $out_matrix \
                        --precision fp64 \
                        --umsource $UNIMORPH \
                        --src_lang $src --trg_lang $trg \
                        --out_dir $output_dir \
                        $oov_args"

                    if [[ $7 == 0 ]]; then
    #                    python3 "$ROOT/src/test/eval_translation_dev.py" --filter_dict ";" --dictionary $dev_dict --test_info $dev_test_info $args1 $args2 $args3 $args4 $oov_args > "$dev_res_file"
    #                    python3 "$ROOT/src/test/eval_translation_dev.py" --filter_dict "ADJ"   --dictionary $test_dict --test_info $test_test_info $args1 $args2 $args3 $args4 $oov_args > "$test_res_file" #$oov_args # > "$dev_res_file"
                        # echo -n "  - SGD reinfl + analysis |  Dev Translation: "
                        # python3 "$ROOT/src/test/eval_translation_dev.py"  ${eval_args} \
                        #     --dictionary $dev_dict \
                        #     --test_info $dev_test_info \
                        #     --reinflection_models_dir $rm_source/$src-$trg \
                        #     --analysis_models_dir $am_source/$src-$trg | tee $dev_res_file

                        echo -n "  - SGD reinfl + analysis |  Test Translation: "
                        python3 "$ROOT/src/test/eval_translation_dev.py"  ${eval_args} \
                            --dictionary $test_dict \
                            --test_info $test_test_info \
                            --reinflection_models_dir $rm_source/$src-$trg \
                            --analysis_models_dir $am_source/$src-$trg | tee $test_res_file 
                    elif [[ $7 == 1 ]]; then
                        # echo -n "  - SGD reinfl + analysis (hybrid) |  Dev Translation: "
                        # python3 "$ROOT/src/test/eval_translation_dev.py"  ${eval_args} \
                        #     --hybrid  \
                        #     --dictionary $dev_dict \
                        #     --test_info $dev_test_info \
                        #     --reinflection_models_dir $rm_source/$src-$trg \
                        #     --analysis_models_dir $am_source/$src-$trg | tee $dev_res_file
                        
                        echo -n "  - SGD reinfl + analysis (hybrid) |  Test Translation: "
                        python3 "$ROOT/src/test/eval_translation_dev.py"  ${eval_args} \
                            --hybrid  \
                            --dictionary $test_dict \
                            --test_info $test_test_info \
                            --reinflection_models_dir $rm_source/$src-$trg \
                            --analysis_models_dir $am_source/$src-$trg | tee $test_res_file
                    elif [[ $7 == 2 ]]; then
                    #     echo -n "  - SGD reinfl (no analysis) |  Dev Translation: "
                    #     python3 "$ROOT/src/test/eval_translation_dev.py"  ${eval_args} \
                    #         --dictionary $dev_dict \
                    #         --test_info $dev_test_info \
                    #         --reinflection_models_dir $rm_source/$src-$trg | tee $dev_res_file

                        echo -n "  - SGD reinfl (no analysis) |  Test Translation: "
                        python3 "$ROOT/src/test/eval_translation_dev.py"  ${eval_args} \
                            --dictionary $test_dict \
                            --test_info $test_test_info \
                            --reinflection_models_dir $rm_source/$src-$trg | tee $test_res_file
                            # no analysis argument
                    fi
                done
            fi

        done
        echo
    done
    echo
done
