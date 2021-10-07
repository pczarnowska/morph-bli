import sys
import collections
from src.utils.cupy_utils import *
import numpy as np
import pickle
from src.utils.tag_utils import *

from src.test.helpers import get_common_lemmas, get_common_tags
import os

reinfl_path = os.path.abspath(__file__ + "/../../../../reinflector/src")
sys.path.append(reinfl_path)
from sig_decode import reinflect, analyse
from util import edit_distance


def get_all_words_for_lemma_eval(opts, src, src2trg, srctrg2info):
    srcd, trgd = opts.src_data, opts.trg_data

    source_lemmas = []
    source_forms = []
    target_lemmas = []
    target_forms = []
    morphology_source = []
    morphology_target = []

    for src_ind in src:
        srcw = srcd.words[src_ind]

        # finding all good targets and their lemmas
        all_good_targets = src2trg[src_ind]
        tforms, tlemmas, slemmas, src_morph, trg_morph = [], [], [], [], []
        for t in all_good_targets:
            tw = trgd.words[t]
            m, sl, tl = srctrg2info[(srcw, tw)][0]

            if isinstance(m, tuple):
                m_s, m_t = m
            else:
                m_s = m
                m_t = m

            tforms.append(tw)
            tlemmas.append(tl)
            slemmas.append(sl)
            src_morph.append(m_s)
            trg_morph.append(m_t)

        source_forms.append(srcw)
        target_forms.append(tforms)
        target_lemmas.append(tlemmas)
        source_lemmas.append(slemmas)
        morphology_source.append(src_morph)
        morphology_target.append(trg_morph)

    return source_forms, source_lemmas, target_forms, target_lemmas,\
        morphology_source, morphology_target


def get_unique_lemmas_from_dict(opts, all_src_lemmas, morphology, x):
    """
    This function is used to chose one lemma and one tag from the given
    list of source lemmas and corresponding tags. The first suitable lemma
    is chosen.
    :param opts:
    :param all_src_lemmas:
    :param morphology:
    :return:
    """
    srcd, trgd = opts.src_data, opts.trg_data

    chosen_src_lemmas = []
    chosen_morph = []

    unknown_lemmas = 0
    for i, source_word_lemmas in enumerate(all_src_lemmas):
        src_lem, morph = None, None
        second_best, sb_morph = None, None

        # just take the first lemma in the vocabulary
        for j, lem in enumerate(source_word_lemmas):
            if lem not in srcd.word2ind:
                continue

            indx = srcd.word2ind[lem]
            if indx >= srcd.first_oov_index:
                pass
            else:
                src_lem = lem
                morph = morphology[i][j]
                break

        if not src_lem and not second_best:
            chosen_morph.append("?")
            chosen_src_lemmas.append("?")
            unknown_lemmas += 1
        else:
            if not src_lem:
                # print("OOV lemma")
                src_lem = second_best
                morph = sb_morph
            chosen_morph.append(morph)
            chosen_src_lemmas.append(src_lem)

    return chosen_src_lemmas, chosen_morph, unknown_lemmas


def get_all_target_morphs(args, trgd, src_morphs):
    # used for unrelated languages - get all tags in the target language
    # which match the source
    all_morphs = set()
    for w, morphs in trgd.word2morph.items():
        all_morphs = all_morphs.union(set(morphs))

    return [get_target_morphs(args, smorph, all_morphs)
            for smorph in src_morphs]


def check_morph(src_morph, potential):
    def is_inf(d):
        return ("Finiteness", "nfin") in d.items()

    m1_ftype2val = get_ftype2val(src_morph)
    m2_ftype2val = get_ftype2val(potential)

    if is_inf(m1_ftype2val) ^ is_inf(m2_ftype2val):
        return False

    for (ftype, val_m1) in m1_ftype2val.items():
       if ftype in m2_ftype2val:
           val_m2 = m2_ftype2val[ftype]
           if val_m1 != val_m2:
                return False
    return True


spec_morph = [
    {"PRS;SBJV;V", "NFIN;V", "3;PL;PRS;V", "2;PL;PRS;V",
     "1;PL;PRS;V", "1;PRS;SG;V", "2;PRS;SG;V", "2;IMP;V"},
    {"PST;SBJV;V", "PST;V"}]


def get_target_morphs(args, src_morph, all_morphs):
    if src_morph == "NOTUM":
        return ["NOTUM"]

    full_set = {src_morph}
    if args.src_lang == "eng":
        special = spec_morph
    else:
        special = []

    for s in special:
        if src_morph in s:
            full_set = full_set.union(s)

    good_translations = set()
    for morph in full_set:
        for potential in all_morphs:
            if check_morph(morph, potential):
                good_translations.add(potential)
    return list(good_translations)


def get_translations_via_lemma(
        args, opts, x, z, src, src2trg, good_inds, srctrg2info,
        chunk_size, xp, dtype, get_all_inflections=False):
    srcd, trgd = opts.src_data, opts.trg_data

    src_forms, src_lemmas, trg_forms, trg_lemmas, morphs_src, morphs_trg =\
        get_all_words_for_lemma_eval(opts, src, src2trg, srctrg2info)

    if args.analysis_model:
        print("Applying both the analysis model and the reinflection model")
        chosen_src_lems, chosen_morphs =\
            analyse(args.analysis_model, src_forms)
        chosen_src_lem_inds =\
            get_forms_ids(args, None, chosen_src_lems, srcd, max_id=len(x))
        unknown_lemmas =\
            sum([1 if ind == 0 else 0 for ind in chosen_src_lem_inds])

        print(f"acc for source lemmas: {sum([chosen_src_lems[i] in src_lemmas[i] for i in range(len(src))])/len(src)}")
        print(f"acc for source tags: {sum([chosen_morphs[i] in morphs_src[i] for i in range(len(src))])/len(src)}")

        if args.hybrid2:  # relative + accuracy
            print("HYBRID MODEL 2")
            lem_form_zip = zip(chosen_src_lem_inds, src)
            for i, (lem_id, form_id) in enumerate(lem_form_zip):
                if lem_id == 0 or form_id < lem_id or form_id < 25000:
                    chosen_src_lem_inds[i] = form_id
                    chosen_morphs[i] = "NFIN"
        elif args.hybrid3:  # just accuracy
            print("HYBRID MODEL  3")
            lem_form_zip = zip(chosen_src_lem_inds, src)
            for i, (lem_id, form_id) in enumerate(lem_form_zip):
                if lem_id == 0 or form_id < 25000:
                    chosen_src_lem_inds[i] = form_id
                    chosen_morphs[i] = "NFIN"
        elif args.hybrid:
            print("HYBRID MODEL")
            lem_form_zip = zip(chosen_src_lem_inds, src)
            for i, (lem_id, form_id) in enumerate(lem_form_zip):
                if lem_id == 0 or form_id < lem_id:
                    chosen_src_lem_inds[i] = form_id
                    # this means the resulting trans. wont be inflected
                    chosen_morphs[i] = "NFIN"
    else:
        print("Oracle analyzer")
        chosen_src_lems, chosen_morphs, unknown_lemmas =\
            get_unique_lemmas_from_dict(opts, src_lemmas, morphs_src, x)
        chosen_src_lem_inds =\
            [srcd.word2ind[src_lem] for src_lem in chosen_src_lems]

    lem_translation_results = get_normal_translations(
        args, x, z, chosen_src_lem_inds, good_inds, chunk_size, xp, dtype)

    lemma_translations = [trgd.words[lem_translation_results[src_ind]]
                          for src_ind in chosen_src_lem_inds]

    if get_all_inflections or morphs_src != morphs_trg:
        average_prec = 0
        average_rec = 0
        total_correct_lems = 0

        # chosen morphs are for the source and we need to find sets of target
        # tags that match the source tags
        all_target_morphs = get_all_target_morphs(args, trgd, chosen_morphs)
        print("chosen morphs")
        print(chosen_morphs[:10], all_target_morphs[:10])

        reinflected_translations = reinflect(
            args.reinflection_model, lemma_translations,
            all_target_morphs, multi=True)
        print("\n======================")
        print("RESULTS:")
        print("SRC forms:", src_forms[:15])
        print("SRC lemmas:", chosen_src_lems[:15])
        print("TAGS:", chosen_morphs[:15])
        print("\n")
        print("Lemma translations:", lemma_translations[:15])
        print("Reinflected:", reinflected_translations[:15])

        print(f"acc lemmas: {sum([lemma_translations[i] in trg_lemmas[i] for i in range(len(src))])/len(src)}")

        for i in range(len(src)):
            # we are only concerned with correct lemmas
            if lemma_translations[i] not in trg_lemmas[i]:
                continue
            total_correct_lems += 1

            result_trans = list(set(reinflected_translations[i]))

            expected_forms = []
            for tf in trg_forms[i]:
                lems = trgd.word2lemma[tf]
                if lemma_translations[i] in lems:
                    expected_forms.append(tf)
            expected_forms = set(expected_forms)

            correct_infls = sum(
                [trans in expected_forms for trans in result_trans])
            precision = correct_infls/len(result_trans)
            recall = correct_infls/len(expected_forms) if expected_forms else 0

            average_prec += precision
            average_rec += recall

            if i < 6:
                print(f"source word {src_forms[i]}: p {precision} r {recall}")
                print("expected", expected_forms)
                print("translated", result_trans)

        print(f"Uknown lemmas: {unknown_lemmas} our of {len(src)}")
        print("======================\n")

        if total_correct_lems != 0:
            mac_prec = average_prec/total_correct_lems
            mac_recall = average_rec/total_correct_lems

            print("RESULT: AVERAGE precision and average recall:",
                  mac_prec, mac_recall)

        forms_ids = get_forms_ids(
            args, all_target_morphs, reinflected_translations, trgd)
        return dict(zip(src, forms_ids))

    else:
        reinflected_translations =\
            reinflect(args.reinflection_model, lemma_translations, chosen_morphs)

        print("\n======================")
        print("RESULTS:")
        print("SRC forms:", src_forms[:7])
        print("SRC lemmas:", chosen_src_lems[:7])
        print("TAGS:", chosen_morphs[:7])
        print("\n")
        print("Lemma translations:", lemma_translations[:7])
        print("Reinflected:", reinflected_translations[:7])

        print(f"acc lemmas: {sum([lemma_translations[i] in trg_lemmas[i] for i in range(len(src))])/len(src)}")
        print(f"acc forms: {sum([reinflected_translations[i] in trg_forms[i] for i in range(len(src))])/len(src)}")
        print(f"Uknown lemmas: {unknown_lemmas} our of {len(src)}")
        print("======================\n")

        cor_lem = 0
        mess_up = 0
        messed_up_forms = collections.Counter()
        for i in range(len(src)):
            lem_ok = lemma_translations[i] in trg_lemmas[i]
            form_ok = reinflected_translations[i] in trg_forms[i]

            if lem_ok:
                cor_lem += 1
            if lem_ok and not form_ok:
                mess_up += 1
                messed_up_forms[chosen_morphs[i]] += 1
        print("Mess up: ", mess_up, " out of", cor_lem)
        print(messed_up_forms)

        forms_ids = get_forms_ids(args, None, reinflected_translations, trgd)
        return dict(zip(src, forms_ids))


def get_forms_ids(args, tags, reinflected_translations, dat, max_id=0):
    """
    Turn the translation into IDs. If min_distance argument is true then
    the translations which are not in the target emb. matrix are turned
    into a word in the vocab with min edit distance with them.
    :param args:
    :param reinflected_translations:
    :param trgd:
    :return:
    """
    forms_ids = []
    missing = 0

    flet2words = collections.defaultdict(set)
    for w in dat.words:
        if dat.word2ind[w] < max_id or max_id == 0:
            # this is a tiny bit hacky optimisation
            flet2words[w[0]].add(w)

    for i, wform_m in enumerate(reinflected_translations):
        if isinstance(wform_m, list):
            wform = None
            for j, x in enumerate(tags[i]):
                if x == "NFIN;V":
                    wform = wform_m[j]
            if not wform:
                wform = sorted(
                    wform_m,
                    # get the most frequent inflection of the resulting ones
                    key=lambda x: 100000000 if x not in dat.word2ind \
                        else dat.word2ind[x])[0]
        else:
            wform = wform_m

        if wform in dat.word2ind and (
                max_id == 0 or dat.word2ind[wform] < max_id):
            forms_ids.append(dat.word2ind[wform])
            continue

        missing += 1
        if not args.min_distance:
            forms_ids.append(0)
        else:
            flet_words = flet2words[wform[0]]
            closest_word = min(
                flet_words, key=lambda x: edit_distance(x, wform))
            if missing < 20:
                print("closest to ", wform, closest_word)
            forms_ids.append(dat.word2ind[closest_word])

    print("Translated forms missing in embs:", missing)
    return forms_ids


def get_normal_translations(args, x, z, src, good_inds, chunk_size, xp, dtype):
    global BIG_CACHE

    # Find translations
    translation = collections.defaultdict(int)

    if args is None or args.retrieval == 'nn':  # Standard nearest neighbor
        for i in range(0, len(src), chunk_size):
            j = min(i + chunk_size, len(src))

            similarities = x[src[i:j]].dot(z.T)
            if good_inds:
                ginds = good_inds[(i, j)]
                mask = np.full((j - i, z.shape[0]), -2, dtype)
                xs, ys = ginds
                mask[xs, ys] = 0
                similarities = similarities + mask
            nn = similarities.argmax(axis=1).tolist()

            for k in range(j - i):
                res = nn[k]
                translation[src[i + k]] = res

    elif args.retrieval == 'invnn':  # Inverted nearest neighbor
        best_rank = np.full(len(src), x.shape[0], dtype=int)
        best_sim = np.full(len(src), -100, dtype=dtype)
        for i in range(0, z.shape[0], chunk_size):
            j = min(i + chunk_size, z.shape[0])
            similarities = z[i:j].dot(x.T)
            ind = (-similarities).argsort(axis=1)
            ranks = asnumpy(ind.argsort(axis=1)[:, src])
            sims = asnumpy(similarities[:, src])
            for k in range(i, j):
                for l in range(len(src)):
                    rank = ranks[k - i, l]
                    sim = sims[k - i, l]
                    if rank < best_rank[l] or (
                                    rank == best_rank[l] and sim >
                                best_sim[l]):
                        best_rank[l] = rank
                        best_sim[l] = sim
                        translation[src[l]] = k
    elif args.retrieval == 'invsoftmax':  # Inverted softmax
        sample = xp.arange(
            x.shape[
                0]) if args.inv_sample is None else \
            xp.random.randint(0, x.shape[0], args.inv_sample)
        partition = xp.zeros(z.shape[0])
        for i in range(0, len(sample), chunk_size):
            j = min(i + chunk_size, len(sample))
            partition += xp.exp(
                args.inv_temperature * z.dot(x[sample[i:j]].T)).sum(
                axis=1)
        for i in range(0, len(src), chunk_size):
            j = min(i + chunk_size, len(src))
            p = xp.exp(
                args.inv_temperature * x[src[i:j]].dot(z.T)) / partition
            nn = p.argmax(axis=1).tolist()
            for k in range(j - i):
                translation[src[i + k]] = nn[k]
    return translation


def get_map_per_word(ranked_scores, infl_ids):
    good_so_far = 0
    p_map = {inflid: 0 for inflid in infl_ids}
    ap = 0
    for i in range(len(ranked_scores)):
        if ranked_scores[i] in infl_ids:
            good_so_far += 1
            prec = good_so_far/(i+1)
            p_map[ranked_scores[i]] = prec
            ap += prec
            if good_so_far == len(infl_ids):
                break
    return ap/len(infl_ids), p_map


def same_paradigm(trgd, w1, w2, pos):
    posok1 = any([pos in morph for morph in trgd.word2morph[w1]])
    posok2 = any([pos in morph for morph in trgd.word2morph[w2]])

    if posok1 and posok2 and same_lemma(trgd, w1, w2):
        return True
    else:
        return False


def same_lemma(trgd, w1, w2, trans_lemmas=[]):
    lem1 = trgd.word2lemma[w1]
    lem2 = trgd.word2lemma[w2]

    if trans_lemmas:
        return len(lem2.intersection(trans_lemmas))

    union = lem1.intersection(lem2)
    return len(union) > 0


# this is only called for source words which had correct inflection returned
def get_map(
        args, opts, x, z, src, translations, src2trg, srctrg2info, chunk_size):
    srcd, trgd = opts.src_data, opts.trg_data
    ap_map = collections.defaultdict(int)

    for i in range(0, len(src), chunk_size):
        j = min(i + chunk_size, len(src))

        similarities = x[src[i:j]].dot(z.T)
        # only take the top 100000 and set precision to all words below to 0
        ranked_scores = (-similarities).argsort(axis=1)

        for k in range(j - i):
            srcind = src[i + k]

            # only evaluate for correct translations
            srcw = srcd.words[srcind]

            if translations[srcind] in src2trg[srcind]:
                trgw = trgd.words[translations[srcind]]

                trans_lemmas = set()
                for (m, sl, tl) in srctrg2info[(srcw, trgw)]:
                    trans_lemmas.add(tl)

                print("\npair:", srcw, trgw)

                inflid2word = {}
                inflid2morphs = collections.defaultdict(set)
                for t in src2trg[srcind]:
                    tw = trgd.words[t]

                    if same_lemma(trgd, trgw, tw, trans_lemmas):

                        morphs = set()
                        for (m, sl, tl) in srctrg2info[(srcw, tw)]:
                            morphs.add(m)

                        inflid2word[t] = tw
                        inflid2morphs[t] = inflid2morphs[t].union(morphs)

                print(inflid2word)

                ap, p_map = get_map_per_word(
                    ranked_scores[k], list(inflid2word.keys()))
                ap_map[srcw] = (ap, p_map, inflid2word, inflid2morphs)

                print("AP score:", ap)
                for infl, p in p_map.items():
                    print(inflid2word[infl], inflid2morphs[infl], p)

    if len(ap_map) != 0:
        mean_ap = sum([v for k, (v, _, _, _) in ap_map.items()])/len(ap_map)
    else:
        mean_ap = 0

    pickle_file = open(
        f"{args.out_dir}/pickled_map_eval_{len(x)}_{args.test_info}.txt", "wb")
    pickle.dump(ap_map, pickle_file)
    pickle_file.close()

    print("MAP SCORE:", len(x),  mean_ap)
    return ap_map, mean_ap


def rate_translation(
        src, src2trg, translation, opts, unimorph_only=False, out_file=None):
    # Compute accuracy
    srcd, trgd = opts.src_data, opts.trg_data

    t_results, m_results, l_results = [], [], []
    out_for_analysis = ""
    for i in src:
        tres = mres = lres = 0
        srcw = srcd.words[i]
        trgw = trgd.words[translation[i]]

        if translation[i] in src2trg[i]:
            tres = 1
        else:
            gold_twords = [trgd.words[x] for x in src2trg[i]]
            try:
                if trgw in trgd.word2morph:
                    if get_common_tags(opts, srcw, trgw):
                        mres = 1
                    if get_common_lemmas(gold_twords, trgw, trgd.word2lemma):
                        lres = 1

                    golt_twords_tags = ' '.join(
                        f"{word} [{' '.join(trgd.word2morph[word])}]"
                        for word in gold_twords)

                    out_for_analysis +=\
                        f"{srcw} - {trgw} [{' '.join(trgd.word2morph[trgw])}]"+\
                        f" ({golt_twords_tags}), MRES:{mres} LRES:{lres}\n"
                else:
                    out_for_analysis +=\
                        f"{srcw} - {trgw} ({' '.join(gold_twords)})\n"
            except:
                pass

        t_results.append(tres)

        if unimorph_only and trgw in trgd.word2morph:
            gold_twords = [trgd.words[goldid] for goldid in src2trg[i]]
            if get_common_tags(opts, srcw, trgw):
                mres = 1
            if get_common_lemmas(gold_twords, trgw, trgd.word2lemma):
                lres = 1
            m_results.append(mres)
            l_results.append(lres)

    accuracy = 0 if not t_results else np.mean(t_results)
    morph_accuracy = 0 if not m_results else np.mean(m_results)
    lemma_accuracy = 0 if not l_results else np.mean(l_results)

    if out_file:
        with open(out_file, "a") as f:
            f.write(out_for_analysis)
    return (accuracy, morph_accuracy, lemma_accuracy)
