#!/usr/bin/env bash

source activate bli

LANGS=( "fra" "spa" "ita" )
DATA=$5
TEST_DATA=$6

for ((i = $1; i < $2; i++))
do
    for ((j = $3; j < $4; j++))
        do

            if [[ $i == $j ]]; then
                continue
            fi

            l1=${LANGS[$i]}
            l2=${LANGS[$j]}

            echo $l1 $l2 $DATA

            ./scripts/my_reinflection.sh hmmfull $l1-$l2 $DATA ${TEST_DATA}
        done
done
