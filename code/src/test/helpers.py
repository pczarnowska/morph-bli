from src.utils.tag_utils import *
from src.utils.commons import *
import src.utils.embeddings as embeddings
import regex as re


def get_best_model_file(models_dir):
    if models_dir is None:
        return None

    print("\n--", models_dir)
    fpaths = get_file_paths(models_dir, in_name="epoch")

    best_acc, best_file = 0, ""
    for fpath in fpaths:
        name = fpath.split("/")[-1]
        m1 = re.search(r"\.acc_([0-9]{2}.[0-9]+)(?=\.)", name)
        m2 = re.search(r"\.acc_lem_([0-9]{2}.[0-9]+)(?=\.)", name)

        m = m2 if m2 else m1
        if m:
            try:
                acc = float(m.group(1))
                if acc > best_acc:
                    best_file = fpath
                    best_acc = acc
            except:
                continue
    print(f"Best found model acc: {best_acc}")
    return best_file


def get_common_tags(opts, srcw, trgw):
    srcd, trgd = opts.src_data, opts.trg_data
    stag = get_tag(srcw, srcd.word2morph, lang=srcd.lang)
    ttag = get_tag(trgw, trgd.word2morph, lang=trgd.lang)
    return set(stag).intersection(set(ttag))


def get_common_lemmas(gold_twords, trgw, trg2lemma):
    glemmas = set()
    for gword in gold_twords:
        glemmas = glemmas.union(set(trg2lemma[gword]))
    tlemmas = set(trg2lemma[trgw])
    return glemmas.intersection(tlemmas)


def print_res(results, latex=False):
    header = ["-"]
    line = []
    first_col = []

    for (name, sample), ((acc, std),
            (macc, mstg), (lacc, lstd), srclen, cov, _) in results.items():
        fname = f"{name} (sample {sample})"
        if lacc == 0 and macc == 0:
            print(f"{fname}: Acc {acc:4.2%}, Cov {cov:4.2%}, Len {srclen}")
        else:
            print(f"{fname}: Acc {acc:4.2%}, Morph {macc:4.2%}, " +
                  f"Lemma {lacc:4.2%}, Cov {cov:4.2%}, Len {srclen}")
        if latex:
            header.append(f"{name}")
            first_col.append("-")
            line.append(f"{acc:4.2%}, {cov:4.2%}, {srclen}".replace("%", "\%"))
    if latex:
        print("\n", generate_latex_table(header, first_col, [line]))


def get_morph_counts(big_dict):
    morph_dat = []

    for morph, d in big_dict.morph2dict.items():
        in_src, _, in_cov = d.get_invocab()
        all_src, _ , all_cov = d.get_all()
        morph_dat.append((morph, len(in_src), in_cov, len(all_src)))
    return morph_dat


def process_accuracies(accuracies):
    accs = np.array(accuracies)
    res = []
    for i in range(accs.shape[1]):
        vals = accs[:, i]
        acc_mean = 0 if not vals else np.mean(vals)
        acc_std = 0 if not vals else np.std(vals)
        res.append((acc_mean, acc_std))
    return res


def turn_tags_to_ids(tags, tag2id):
    tag_ids = []
    for tag_list in tags:
        id_list = [tag2id[tag] if tag in tag2id else -1 for tag in tag_list]
        tag_ids.append(id_list)
    return tag_ids


def retrieve_tags(opts, big_dict):
    srcd, trgd = opts.src_data, opts.trg_data

    srcd.tags = get_tags(srcd.words, srcd.word2morph, lang=srcd.lang)
    trgd.tags = get_tags(trgd.words, trgd.word2morph, lang=trgd.lang)

    tag2ind = {}
    for tag_list in srcd.tags:
        for tag in tag_list:
            tag2ind.setdefault(tag, len(tag2ind))
    for tag_list in trgd.tags:
        for tag in tag_list:
            tag2ind.setdefault(tag, len(tag2ind))
    opts.tag2ind = tag2ind

    srcd.tag_ids = turn_tags_to_ids(srcd.tags, tag2ind)
    trgd.tag_ids = turn_tags_to_ids(trgd.tags, tag2ind)


def get_only_unimorph_data(opts, src_matrix, trg_matrix):
    srcd, trgd = opts.src_data, opts.trg_data

    dats = [srcd, trgd]
    embs = [src_matrix, trg_matrix]

    res = []

    for i, dat in enumerate(dats):
        new_words, new_embs, new_tags = [], [], []
        old2new_map = {}
        for j, tag_list in enumerate(dat.tag_ids):
            if tag_list and j < len(embs[i]):
                old2new_map[j] = len(new_words)
                new_words.append(dat.words[j])
                new_embs.append(embs[i][j])
                new_tags.append(tag_list)

        res.append((new_words, opts.xp.array(new_embs), new_tags, old2new_map))
    return res


################################
#       READING EMBEDDINGS     #
################################

def read_matrix(matrix_source, encoding):
    W = None
    with open(matrix_source, "r", encoding=encoding) as f:
        l = [[float(num) for num in line.split(' ')] for line in f]
        W = np.array(l)
    return W


def get_x_z_with_oovs(original_x_vecs, original_z_vecs, matrix,
                      src_oov_vecs, trg_oov_vecs,
                      normalize, encoding, dtype, threshold):
    src_words, orig_x = embeddings.read(
        original_x_vecs, encoding, dtype, threshold=threshold)
    trg_words, orig_z = embeddings.read(
        original_z_vecs, encoding, dtype, threshold=threshold)
    x, x_means, x_norms = embeddings.normalize(
        orig_x, normalize, return_means_and_norms=True)
    z, z_means, z_norms = embeddings.normalize(
        orig_z, normalize, return_means_and_norms=True)
    x = x.dot(matrix)

    src_oov_words = oov_x = trg_oov_words = oov_z = None
    if src_oov_vecs and trg_oov_vecs:
        try:
            src_oov_words, oov_x = embeddings.read(
                src_oov_vecs, encoding, dtype)
            trg_oov_words, oov_z = embeddings.read(
                trg_oov_vecs, encoding, dtype)
            # normalise in the same way as the original embeddings
            oov_x = embeddings.normalize_oovs(
                oov_x, normalize, x_means, x_norms)
            oov_z = embeddings.normalize_oovs(
                oov_z, normalize, z_means, z_norms)
            oov_x = oov_x.dot(matrix)
        except:
            print("Exception in OOV handling")

    return src_words, trg_words, src_oov_words, \
        trg_oov_words, x, z, oov_x, oov_z


def process_trained_embeddings(args, threshold, dtype):
    def separate(allws, in_vocabws, allx, invocabx):
        if len(allws) > len(in_vocabws):
            vlen = len(in_vocabws)
            assert allws[:vlen] == in_vocabws
            return allws[:vlen], allx[:vlen], allws[vlen:], allx[vlen:]
        else:
            return allws, allx, [], None

    src_oov_words, trg_oov_words, oov_x, oov_z = [], [], None, None

    src_words, x = embeddings.read(args.src_embeddings, args.encoding, dtype,
                                   threshold=threshold)
    trg_words, z = embeddings.read(args.trg_embeddings, args.encoding, dtype,
                                   threshold=threshold)

    if args.original_src_vecs is not None:
        orig_src_words, ox = embeddings.read(
            args.original_src_vecs, args.encoding, dtype)
        orig_trg_words, oz = embeddings.read(
            args.original_trg_vecs, args.encoding, dtype)

        src_words, x, src_oov_words, oov_x =\
            separate(src_words, orig_src_words, x, ox)
        trg_words, z, trg_oov_words, oov_z =\
            separate(trg_words, orig_trg_words, z, oz)

    return src_words, trg_words, src_oov_words, \
        trg_oov_words, x, z, oov_x, oov_z
