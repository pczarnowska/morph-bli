#!/bin/bash

source activate bli

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA="$ROOT/../data"
RESOURCES="$ROOT/../resources"
UNIMORPH="$RESOURCES/unimorph"
FASTTEXT="$RESOURCES/fastText"
MDICTS="$DATA/dictionaries/morph_dictionaries"
WIKT="$RESOURCES/wn_data"

RLCOUNT=3
RLNAMES=('french' 'spanish' 'italian')
RLCODES2=('fra' 'spa' 'ita')
RLCODES1=('fr' 'es' 'it')


# options: bel (belarusian), dsb (lower sorbian)

mkdir -p "$DATA/embeddings/models"
mkdir -p "$DATA/embeddings/original"
mkdir -p "$DATA/dictionaries/muse"
mkdir -p "$RESOURCES"
mkdir -p "$DATA"


export PYTHONPATH=$PYTHONPATH:$ROOT
cd $ROOT

# getting fasttext
if [[ ! -d "$FASTTEXT" ]]; then  
    echo ">> Retrieving fastText source..."
    wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
    unzip v0.9.2.zip
    mv fastText-0.9.2 $FASTTEXT
    cd $FASTTEXT
    make
    cd $ROOT
fi


if [[ ! -d "$WIKT" ]]; then  
    # get extended wordnet
    cd $RESOURCES
    wget http://compling.hss.ntu.edu.sg/omw/wn-wikt.zip
    unzip wn-wikt.zip
    mv data $WIKT

    # get open multilingual wordnet
    wget http://compling.hss.ntu.edu.sg/omw/wns/spa.zip
    wget http://compling.hss.ntu.edu.sg/omw/wns/fra.zip
    wget http://compling.hss.ntu.edu.sg/omw/wns/ita.zip
    unzip spa.zip -d $WIKT/open
    unzip fra.zip -d $WIKT/open
    unzip ita.zip -d $WIKT/open
    rm *.zip
    cd $WIKT/open 
    rm -rf iwn
    mv */wn-data*.tab .
    cd $ROOT
fi

# getting muse dictionaries
for ((i = 0; i < $RLCOUNT; i++))
do
   code1=${RLCODES1[$i]}
   ncode1=${RLCODES2[$i]}
   for ((j = 0; j < $RLCOUNT; j++))
   do
       if [[ $i == $j ]]; then
           continue
       fi

       code2=${RLCODES1[$j]}
       ncode2=${RLCODES2[$j]}

       if [[ ! -f "$DATA/dictionaries/muse/$ncode1-$ncode2.train.txt" ]]; then
            echo ">> Retrieving MUSE dictionaries ${RLNAMES[$i]}-${RLNAMES[$j]}..."

            wget -q "https://dl.fbaipublicfiles.com/arrival/dictionaries/$code1-$code2.0-5000.txt" -O "$DATA/dictionaries/muse/$ncode1-$ncode2.train.txt"
            wget -q "https://dl.fbaipublicfiles.com/arrival/dictionaries/$code1-$code2.5000-6500.txt"  -O "$DATA/dictionaries/muse/$ncode1-$ncode2.test.txt"
            shuf $DATA/dictionaries/muse/$ncode1-$ncode2.train.txt > $DATA/dictionaries/muse/$ncode1-$ncode2.train.shuf.txt
       fi
   done
done


for ((i = 0; i < $RLCOUNT; i++))
do
    code1=${RLCODES1[$i]}
    code2=${RLCODES2[$i]}

    if [[ ! -f "$UNIMORPH/$code2/$code2" ]]; then  
        # Get unimorph
        echo ">> Retrieving unimorph (${RLNAMES[$i]})..."

        url=https://raw.githubusercontent.com/unimorph/$code2/master/$code2
        if [[ `wget -S --spider $url 2>&1 | grep 'HTTP/1.1 200 OK'` ]]; then
            mkdir -p "$UNIMORPH/$code2"
            wget -q $url -O "$UNIMORPH/$code2/$code2"
        fi
    fi
    
    if [[ ! -f "$DATA/embeddings/models/wiki.$code1.bin" ]]; then
        echo ">> Retrieving embeddings for ${RLNAMES[$i]}..."

        # Get fasttext Wikipedia embeddings
        wget -q https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.$code1.zip -O  "$DATA/embeddings/models/$code1-bin-vec.zip"

        unzip "$DATA/embeddings/models/$code1-bin-vec.zip" -d "$DATA/embeddings/models/"
        mv "$DATA/embeddings/models/wiki.$code1.vec" "$DATA/embeddings/original/$code2-fasttext.emb.txt"

        rm "$DATA/embeddings/models/$code1-bin-vec.zip"
        rm "$DATA/embeddings/models/wiki.$code1.vec"
    fi

    if [[ ! -f "$DATA/embeddings/original/$code2-fasttext-oov.emb.txt" ]]; then
        echo ">> Getting oov embeddings (${RLNAMES[$i]})..."

        python3 "$ROOT/src/data_prep/oov_embs.py"  --ftext_source "$FASTTEXT/fasttext" --lang $code2 --ftext_model "$DATA/embeddings/models/wiki.$code1.bin" --ftext_vecs "$DATA/embeddings/original/$code2-fasttext.emb.txt" --out_file "$DATA/embeddings/original/$code2-fasttext-oov.emb.txt" --wnsource $WIKT --umsource $UNIMORPH
    fi

done


# getting morphorogically rich dictionaries from the repo
if [[ ! -d "$MDICTS/${RLCODES2[0]}-${RLCODES2[1]}" ]]; then  
    echo ">> Retrieving morphological dictionaries"
    git clone https://github.com/pczarnowska/morph_dictionaries.git
    mv morph_dictionaries $MDICTS
    cd  $MDICTS
    mv */* .
    rm -r */
fi

