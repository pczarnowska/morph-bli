#./fasttext print-word-vectors model.bin < oov.txt

from subprocess import call, Popen, PIPE
from src.utils.embeddings import read
import argparse
from src.utils.commons import *
import re


def get_oov_vecs(
        fasttext_path, model_path, words_with_embs, new_vocab, out_file, dim):
    """
    :param words_with_embs:
    :param new_vocab:
    :return:
    """
    print(f"Word forms for vocabulary {len(new_vocab)}")
    oovs = new_vocab.difference(words_with_embs)
    oovs = sorted(oovs)
    print(f"Number of OOV words: {len(oovs)}")

    oov_str = " ".join(oovs)
    tmp_file = "tmp.txt"
    fin = open(tmp_file, "w")
    try:
        fin.write(oov_str)
    finally:
        fin.close()

    fin = open(tmp_file, "r")
    fout = open(out_file, 'w')
    try:
        args = [f"{fasttext_path}", "print-word-vectors", f"{model_path}"]
        p = Popen(args, stdin=fin, stdout=PIPE)
        (output, err) = p.communicate()
        output = output.decode("utf-8")
        oov_num = len(output.split("\n")) - 1

        fout.write(f"{oov_num} {dim}\n")
        fout.write(output)
    finally:
        fin.close()
        fout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ftext_source', help='the path to the fasttext model')
    parser.add_argument('--lang')
    parser.add_argument('--ftext_model')
    parser.add_argument('--ftext_vecs')
    parser.add_argument('--out_file', default=None)
    parser.add_argument('--wnsource')
    parser.add_argument('--umsource')
    parser.add_argument('--dim', default=300, help='what is the dimensionality of the embeddings given by the model')
    args = parser.parse_args()

    if not args.out_file:
        args.out_file =\
            f"{vecs_path.rstrip('/').rsplit('/', 1)}/{lang}-oov.emb.txt"

    words, _ = read(args.ftext_vecs)
    print("Read the embeddings")
    concept2forms = read_wn(args.wnsource, args.lang)
    vocab = set([f for _, forms in concept2forms.items() for f in forms])
    print(f"Retrieved wordnet vocabulary {len(vocab)}")

    um_path = get_um_path(args.umsource, args.lang)
    lemma2word, _, _ = read_um(um_path)

    new_vocab = set()
    for v in vocab:
        if v in lemma2word:
            for f in lemma2word[v]:
                if f.count(" ") == 0:
                    f = re.sub(r'[\s\u200e]', '', f)
                    new_vocab.add(f)
    print(f"Retrieved word forms for vocabulary {len(new_vocab)}")

    get_oov_vecs(
        args.ftext_source, args.ftext_model, words,
        new_vocab, args.out_file, args.dim)

if __name__ == "__main__":
    main()
