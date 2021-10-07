import re
import random
import itertools
import os


pos_list = ["adj", "adp", "adv", "art",
            "aux", "clf", "comp", "conj",
            "det", "intj", "n", "num",
            "part", "pro", "propn", "v.cvb", "v.ptcp", "v.msdr", "v"]

feats_to_skip = ['IND', 'ANIM', 'HUM']

# forms with those are likely not to be matched - this is just fyi,
# nothing is done about this in the code
feats_to_ignore = ['INAN', 'FORM', 'NEUT;PL', 'DU', 'PRF;PST']

slav_langs = {"pol", "ces", "rus", "slk", "slv", "ukr", "mkd"}
case_langs = slav_langs
rom_langs = {"fra", "spa", "cat", "por", "ita", "oci"}

# A dictionatry that contains information about
# possible morphological values and their types
um_val2ftype = {}
feat_types = set()


def get_schema():
    with open(UNIMORPH_SCHEMA_SC, "r", errors='surrogateescape') as f:
        for line in f.readlines():
            split = line.strip().split()
            if len(split) == 3:
                feat_type, _, feat = split
                um_val2ftype[feat] = feat_type
                feat_types.add(feat_type)


UNIMORPH_SCHEMA_SC =\
    os.path.abspath(__file__ + "/../../../unimorph_schema.txt")
get_schema()


def get_pos(tag):
    ftype2val = get_ftype2val(tag)
    if "POS" in ftype2val:
        return ftype2val["POS"].upper()
    else:
        return None


def get_ftype2val(tag_str):
    ftype2val = {}
    for val in tag_str.lower().split(";"):
        val = val.strip()
        if val in um_val2ftype:
            ftype = um_val2ftype[val]
            ftype2val[ftype] = val
    if "POS" not in ftype2val:
        return ftype2val

    if ftype2val["POS"] == "v" and \
            ("Finiteness" not in ftype2val or ftype2val["Finiteness"] != "nfin"):
        if "Mood" not in ftype2val:
            ftype2val["Mood"] = "ind"
        if "Tense" not in ftype2val and ftype2val["Mood"] == "ind":
            ftype2val["Tense"] = "prs"
    return ftype2val


def mapping_function(tag, lang):
    """
    Fixes some tag mistmach issues in different languages
    :param tag:
    :param lang:
    :return:
    """
    for fskip in feats_to_skip:
        tag = tag.replace(f"{fskip};", "")
        tag = tag.replace(f";{fskip}", "")

    if get_pos(tag) != "V":
        return [tag]

    if lang == "pol":
        # add missing FEM feature to Polish past and cond forms
        m = re.match(r"^([1-3];COND);(PL;V)(?:$|\n)", tag)
        m2 = re.match(r"^([1-3]);(PL;PST;V)(?:$|\n)", tag)
        pretag = tag
        if m:
            tag = f'{m.group(1)};FEM;{m.group(2)}'
        elif m2:
            tag = f'{m2.group(1)};FEM;{m2.group(2)}'
        return [tag]

    if lang == "slk":
        if ";PRF;" in tag:
            tag = tag.replace("PRF;", "")

    if lang == "rus" or lang == "ukr":
        if "PST" in tag:
            if "PL" in tag:
                return [f"{x};{g};{tag}" for x in
                        range(1, 4) for g in ["FEM", "MASC"]]
            elif "FEM" in tag or "MASC" in tag:
                return [f"{x};{tag}" for x in range(1, 4)]
            else:
                tag = f"3;{tag}"

    if lang == "eng":
        if tag == "NFIN;V" or tag == "V;NFIN":
            tags = [tag]
            for n in range(1, 4):
                if n != 3:
                    tags.append(f"{n};PRS;SG;V")
                tags.append(f"{n};PL;PRS;V")
            tags.append("2;IMP;V")
            tags.append("PRS;SBJV;V")
        elif tag == "PST;V" or tag == "V;PST":
            tags = [tag, "PST;SBJV;V"]
        else:
            return [tag]
        return tags
    return [tag]


def crop_tag(full_tag, supported_feats=None):
    cropped_tag = []

    for val in full_tag.lower().split(";"):
        if not supported_feats:
            cropped_tag.append(val)

        elif (val in um_val2ftype and um_val2ftype[val] in supported_feats):
            cropped_tag.append(val)

    # impose the same order of tag features for all languages
    tag = ';'.join(sorted(cropped_tag))
    return tag


def get_tag(word, unimorph, morph_feats=None, tag2ind=None,
            single_tag=False, lang=None):
    # TEST FUNCTIONS
    def get_likely_tag(tag_list):
        # a test function that can be used if single_tag==True
        rverb = re.compile("(;V$|;V;|^V;)")
        nl = list(filter(rverb.search, tag_list))
        if len(nl) > 0:
            return nl[0]

        rpos = re.compile("(;N$|;N;|^N;)")
        nl = list(filter(rpos.search, tag_list))
        if len(nl) == 1:
            return nl[0]

        rcase = re.compile("(;NOM$|;NOM;|^NOM;)")
        nl = list(filter(rcase.search, nl))
        if len(nl) > 0:
            return nl[0]
        return tag_list[0]

    def get_random_tag(tag_list):
        # a test function that can be used is single_tag==True
        if len(tag_list) == 1:
            return tag_list[0]
        ind = random.randint(0, len(tag_list) - 1)
        return tag_list[ind]

    if word not in unimorph:
        tags = []  # "UNK"
    else:
        tags = set()
        for t in unimorph[word]:
            tags_based_on_tag = [t]
            # use the same tag processing function that
            # was used for dictionary creation
            if lang != None:
                tags_based_on_tag = mapping_function(t, lang)
            for t2 in tags_based_on_tag:
                cropped = crop_tag(t2, morph_feats)
                if not tag2ind:
                    tags.add(cropped)
                elif cropped in tag2ind:
                    tags.add(cropped)
                # return as soon as you get one valid tag
                if single_tag and tags: return list(tags)

    return list(tags)


def get_general_tags(tags):
    all_gen_tags = []
    for tlist in tags:
        general_tags = set()
        for t in tlist:
            S = t.split(";")
            for m in range(2, len(S)):
                for sub in itertools.combinations(S, m):
                    general_tags.add(';'.join(sorted(list(sub))))
        all_gen_tags.append(list[general_tags])
    return all_gen_tags


def get_tags(
        words, unimorph, morph_feats=None,
        tag2ind=None, single_tag=False, lang=None):
    return [get_tag(
        word, unimorph, morph_feats, tag2ind,
        single_tag, lang) for word in words]


def get_lemmas(words, unimorph):
    return [unimorph[word] for word in words]


def get_tag2ind_from_dictionary(opts, train_dictionary):
    f = open(
        train_dictionary, encoding=opts.encoding, errors='surrogateescape')
    tag2ind = {}
    for line in f:
        splits = line.strip().split("\t")
        if len(splits) == 5:
            src, trg, _, _, full_tag = splits
            # retain only relevant tag information (also lang=None because
            # we don't need to process tags coming from dictionary - they
            # should already be processed)
            cropped_tag = crop_tag(full_tag, opts.morph_feats)
            tag2ind.setdefault(cropped_tag, len(tag2ind))
        else:
            print(splits)
    return tag2ind


def get_tag2ind_from_vocabulary(opts):
    tag2ind = {}
    for tag_list in opts.src_data.tags + opts.trg_data.tags:
        for tag in tag_list:
            tag2ind.setdefault(tag, len(tag2ind))
    return tag2ind
