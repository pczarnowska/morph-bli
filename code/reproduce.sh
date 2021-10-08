# This script has hard-coded example runs for the fra-ita language pair.
# One can uncomment selected commands to run experiments for all language pairs.

# ===== SETUP ENVIRONMENT
cd ..
conda env create -f environment.yml
source activate bli
# or another appropriate command from https://pytorch.org/get-started/previous-versions/
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install tabulate pywikibot sklearn wikitextparser lap tqdm pandas
cd code

# ===== GET DATA
./get_morph_data.sh

# ===== TRAIN THE TRANSLATORS
# args: src_start_idx, src_end_idx, trg_start_idx,
#       trg_end_idx, model_start_idx, model_end_idx

# fra-ita (all models)
./tacl.sh 0 1 2 3 0 3

# Check results:
echo "artetxe2016"
cat ../coling/fra-ita/morph_dictionaries/artetxe2016/coling_test-results-morph_dictionaries.txt
echo "ruder2018"
cat ../coling/fra-ita/morph_dictionaries/ruder2018/coling_test-results-morph_dictionaries.txt
echo "sgd (no transducers)"
cat ../coling/fra-ita/morph_dictionaries/sgd/coling_test-results-morph_dictionaries.txt

# all lang pairs (uncomment if needed)
# ./tacl.sh 0 3 0 3 0 3

# ===== GET REINFLECTION AND ANALYSIS DATA
cd ../reinflector
./get_analyze_data.sh analyze_data analyze_data_um
./get_reinfl_data.sh reinfl_data reinfl_data_um

# ===== TRAIN TRANSDUCERS

# for fra-ita (example)
./train_reinfl.sh 0 1 2 3 reinfl_data_um reinfl_data_um
./train_analyzer.sh 0 1 2 3 analyze_data_um analyze_data_um

# for all languages (uncomment if needed)
# .train_reinfl.sh 0 3 0 3 reinfl_data_um reinfl_data_um
# .train_analyzer.sh 0 3 0 3 analyze_data_um analyze_data_um

# ====== EVALUATE THE FULL MODEL (BASIC)
# args: src_start_idx, src_end_idx, trg_start_idx,
#       trg_end_idx, dict_start_idx, dict_end_idx,
#       model_type (0=basic, 1=hybrid, 2=oracle)

cd ../code
# fra-ita (example)
./morph_evaluate.sh 0 1 2 3 0 1 0

# all languages (uncomment if needed)
# ./morph_evaluate.sh 0 3 0 3 0 1 0


# ====== EVALUATE THE FULL MODEL (HYBRID)
# fra-ita (example)
./morph_evaluate.sh 0 1 2 3 0 1 1

echo "sgd (both transducers)"
cat ../coling/fra-ita/morph_dictionaries/sgd/test-results-morph_dictionaries-model_reinfl_reinfl_data_um-model_analyze_analyze_data_um.txt 
echo "sgd (hybrid)"
cat ../coling/fra-ita/morph_dictionaries/sgd/test-results-morph_dictionaries-hybrid-model_reinfl_reinfl_data_um-model_analyze_analyze_data_um.txt 
echo "sgd (oracle)"
cat ../coling/fra-ita/morph_dictionaries/sgd/test-results-morph_dictionaries-model_reinfl_reinfl_data_um-noana.txt 

# all languages (uncomment if needed)
# ./morph_evaluate.sh 0 3 0 3 0 1 0
