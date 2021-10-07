

def create_full_tag_mask(y_tags, tag_count, n_rows):
    masks = [[] for _ in range(tag_count)]
    for z in range(n_rows):
        tag_list = y_tags[z]
        if not tag_list:
            continue
        else:
            for tag in tag_list:
                masks[tag].append(z)
    return masks


def _get_good_inds(x_tags, y_tags, tag_count, chunk_size):
    inds_dict = {}  # a set of tag masks for every chunk
    target_masks = create_full_tag_mask(y_tags, tag_count, len(y_tags))

    # is and js are exaclty as those used in mathing
    for i in range(0, len(x_tags), chunk_size):
        j = min(len(x_tags), i + chunk_size)
        xs, ys = [], []
        for z in range(i, j):
            tag_list = x_tags[z]
            if tag_list:
                for tag in tag_list:
                    yss = target_masks[tag]
                    xss = [z-i] * len(yss)
                    xs += xss
                    ys += yss
        inds_dict[(i, j)] = (xs, ys)
    return inds_dict


def retrieve_good_inds(
        src, max_trg_index, src_tags, trg_tags, chunk_size, opts):
    x_tags = [src_tags[i] for i in src]
    y_tags = trg_tags[:max_trg_index]
    return _get_good_inds(x_tags, y_tags, len(opts.tag2ind), chunk_size)


def get_good_inds_only_lemma(src, max_trg_ind, chunk_size, opts):
    srcd, trgd = opts.src_data, opts.trg_data

    inds_dict = {}  # a set of tag masks for every chunk
    print("filter fun..")
    yss = []
    for i, trg in enumerate(trgd.words):
        if i > max_trg_ind:
            break
        if opts.filter_fun(trg):
            yss.append(i)

    # is and js are exaclty as those used in mathing
    for i in range(0, len(src), chunk_size):
        j = min(len(src), i + chunk_size)
        xs, ys = [], []
        for z in range(i, j):
            xss = [z - i] * len(yss)
            xs += xss
            ys += yss
        inds_dict[(i, j)] = (xs, ys)
    return inds_dict


def get_good_inds_lemma_based(
        src, src2trgs, srctrg2info, max_trg_ind, chunk_size, opts):
    # to enforce the lexeme
    srcd, trgd = opts.src_data, opts.trg_data

    inds_dict = {}  # a set of tag masks for every chunk

    # is and js are exaclty as those used in mathing
    for i in range(0, len(src), chunk_size):
        j = min(len(src), i + chunk_size)
        xs, ys = [], []
        for z in range(i, j):
            src_ind = src[z]
            trgs = src2trgs[src_ind]
            for trg in trgs:
                info_list = srctrg2info[(srcd.words[src_ind], trgd.words[trg])]
                for info in info_list:
                    _, _, trg_lemma = info
                    trg_word_forms = set(trgd.lemma2words[trg_lemma])
                    cand_inds = [trgd.word2ind[w] for w in trg_word_forms
                                 if w in trgd.word2ind and trgd.word2ind[
                                     w] < max_trg_ind]
                    yss = cand_inds
                    xss = [z - i] * len(yss)
                    xs += xss
                    ys += yss
        inds_dict[(i, j)] = (xs, ys)
    return inds_dict


def merge_good_inds(inds1, inds2):
    if inds1 is None:
        return inds2
    if inds2 is None:
        return inds1

    result = {}
    for range in inds1:
        if range not in inds2:
            print("Couldn't merge good inds -- range keys do not match. " +
                  "Returning inds1.")
            return inds1
        xs1, ys1 = inds1[range]
        xs2, ys2 = inds2[range]
        i1_set = set(zip(xs1, ys1))
        i2_set = set(zip(xs2, ys2))
        intersect = i1_set.intersection(i2_set)
        result[range] = [list(x) for x in zip(*intersect)]
    return result









