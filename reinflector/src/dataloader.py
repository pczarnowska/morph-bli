import torch
from reader import  UnimorphReader
import os
import sys
import random

current_dir = os.getcwd()
sys.path.append(f'{current_dir}/../code')
from src.utils.tag_utils import *

BOS = '<s>'
EOS = '<\s>'
PAD = '<PAD>'
UNK = '<UNK>'
EOF = '<eof>'

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
EOF_IDX = 4

class Dataloader(object):
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")


class Seq2SeqDataLoader(Dataloader):
    def __init__(self, train_file, dev_file, test_file=None):
        super().__init__()
        # assert os.path.isfile(train_file)
        # assert os.path.isfile(dev_file)
        # assert test_file is None or os.path.isfile(test_file)
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.batch_data = dict()
        self.nb_train, self.nb_dev, self.nb_test = 0, 0, 0
        self.nb_attr = 0
        self.source, self.target = self.build_vocab()
        self.source_vocab_size = len(self.source)
        self.target_vocab_size = len(self.target)
        if self.nb_attr > 0:
            self.source_c2i = {
                c: i
                for i, c in enumerate(self.source[:-self.nb_attr])
            }
            self.attr_c2i = {
                c: i + len(self.source_c2i)
                for i, c in enumerate(self.source[-self.nb_attr:])
            }
        else:
            self.source_c2i = {c: i for i, c in enumerate(self.source)}
            self.attr_c2i = None
        self.target_c2i = {c: i for i, c in enumerate(self.target)}
        self.sanity_check()

    def sanity_check(self):
        assert self.source[PAD_IDX] == PAD
        assert self.target[PAD_IDX] == PAD
        assert self.source[BOS_IDX] == BOS
        assert self.target[BOS_IDX] == BOS
        assert self.source[EOS_IDX] == EOS
        assert self.target[EOS_IDX] == EOS
        assert self.source[UNK_IDX] == UNK
        assert self.target[UNK_IDX] == UNK

    def build_vocab(self):
        src_set, trg_set = set(), set()
        cnts = []
        for fp in self.train_file:
            cnt = 0
            for src, trg in self.read_file(fp):
                cnt += 1
                src_set.update(src)
                trg_set.update(trg)
            cnts.append(cnt)
        self.nb_train = cnts[0]
        self.nb_dev = sum([1 for _ in self.read_file(self.dev_file)])
        if self.test_file is not None:
            self.nb_test = sum([1 for _ in self.read_file(self.test_file)])
        source = [PAD, BOS, EOS, UNK] + sorted(list(src_set))
        target = [PAD, BOS, EOS, UNK] + sorted(list(trg_set))
        return source, target

    def read_file(self, file):
        raise NotImplementedError

    def _batch_helper(self, lst):
        bs = len(lst)
        srcs, trgs = [], []
        max_src_len, max_trg_len = 0, 0
        for _, src, trg in lst:
            max_src_len = max(len(src), max_src_len)
            max_trg_len = max(len(trg), max_trg_len)
            srcs.append(src)
            trgs.append(trg)
        batch_src = torch.zeros(
            (max_src_len, bs), dtype=torch.long, device=self.device)
        batch_src_mask = torch.zeros(
            (max_src_len, bs), dtype=torch.float, device=self.device)
        batch_trg = torch.zeros(
            (max_trg_len, bs), dtype=torch.long, device=self.device)
        batch_trg_mask = torch.zeros(
            (max_trg_len, bs), dtype=torch.float, device=self.device)
        for i in range(bs):
            for j in range(len(srcs[i])):
                batch_src[j, i] = srcs[i][j]
                batch_src_mask[j, i] = 1
            for j in range(len(trgs[i])):
                batch_trg[j, i] = trgs[i][j]
                batch_trg_mask[j, i] = 1
        return batch_src, batch_src_mask, batch_trg, batch_trg_mask

    def _batch_sample(self, batch_size, file):
        if isinstance(file, list):
            key = tuple(sorted(file))
        else:
            key = file
        if key not in self.batch_data:
            lst = list()
            for src, trg in self._iter_helper(file):
                lst.append((len(src), src, trg))
            self.batch_data[key] = sorted(lst, key=lambda x: x[0])

        lst = self.batch_data[key]
        for start in range(0, len(lst), batch_size):
            yield self._batch_helper(lst[start:start + batch_size])

    def train_batch_sample(self, batch_size):
        yield from self._batch_sample(batch_size, self.train_file)

    def dev_batch_sample(self, batch_size):
        yield from self._batch_sample(batch_size, self.dev_file)

    def test_batch_sample(self, batch_size):
        yield from self._batch_sample(batch_size, self.test_file)

    def encode_source(self, sent):
        if sent[0] != BOS:
            sent = [BOS] + sent
        if sent[-1] != EOS:
            sent = sent + [EOS]
        l = len(sent)
        s = []
        for x in sent:
            if x in self.source_c2i:
                s.append(self.source_c2i[x])
            else:
                s.append(self.attr_c2i[x])
        return torch.tensor(s, device=self.device).view(l, 1)

    def decode_source(self, sent):
        if isinstance(sent, torch.Tensor):
            assert sent.size(1) == 1
            sent = sent.view(-1)
        return [self.source[x] for x in sent]

    def decode_target(self, sent):
        if isinstance(sent, torch.Tensor):
            assert sent.size(1) == 1
            sent = sent.view(-1)
        return [self.target[x] for x in sent]

    def train_sample(self):
        for src, trg in self._iter_helper(self.train_file):
            yield (torch.tensor(src, device=self.device).view(len(src), 1),
                   torch.tensor(trg, device=self.device).view(len(trg), 1))

    def dev_sample(self):
        for src, trg in self._iter_helper(self.dev_file):
            yield (torch.tensor(src, device=self.device).view(len(src), 1),
                   torch.tensor(trg, device=self.device).view(len(trg), 1))

    def test_sample(self):
        for src, trg in self._iter_helper(self.test_file):
            yield (torch.tensor(src, device=self.device).view(len(src), 1),
                   torch.tensor(trg, device=self.device).view(len(trg), 1))

    def _iter_helper(self, file):
        for source, target in self.read_file(file):
            src = [self.source_c2i[BOS]]
            for s in source:
                src.append(self.source_c2i.get(s, UNK_IDX))
            src.append(self.source_c2i[EOS])
            trg = [self.target_c2i[BOS]]
            for t in target:
                trg.append(self.target_c2i.get(t, UNK_IDX))
            trg.append(self.target_c2i[EOS])
            yield src, trg


class SIGMORPHON2019Task1(Seq2SeqDataLoader):
    def build_vocab(self):
        char_set, tag_set = set(), set()
        cnts = []
        for fp in self.train_file:
            cnt = 0
            for lemma, word, tags in self.read_file(fp):
                cnt += 1
                char_set.update(lemma)
                char_set.update(word)
                tag_set.update(tags)
            cnts.append(cnt)
        self.nb_train = cnts[0]
        self.nb_dev = sum([1 for _ in self.read_file(self.dev_file)])
        if self.test_file is None:
            self.nb_test = 0
        else:
            self.nb_test = sum([1 for _ in self.read_file(self.test_file)])
        chars = sorted(list(char_set))
        tags = sorted(list(tag_set))
        self.nb_attr = len(tags)
        source = [PAD, BOS, EOS, UNK] + chars + tags
        target = [PAD, BOS, EOS, UNK] + chars
        return source, target

    def read_file(self, file, lowercase=True):
        try:
            if 'train' in file:
             lang_tag = [file.split('/')[-1].split('.')[0].split("-")[1]]
            elif 'dev' in file:
                lang_tag = [file.split('/')[-1].split('.')[0].split("-")[1]]
            elif 'test' in file:
                lang_tag = [file.split('/')[-1].split('.')[0].split("-")[1]]
            else:
                raise ValueError
        except:
            lang_tag = []

        with open(file, 'r', encoding='utf-8') as fp:
            examples = set()
            for line in fp:
                lemma, word, tags = line.strip().split('\t')
                if lowercase:
                    lemma, word = lemma.lower(), word.lower()
                examples.add((lemma, word, tags))

            for ex in examples:
                lemma, word, tags = ex
                yield list(lemma), list(word), lang_tag + tags.split(';')

    def _iter_helper(self, file):
        if not isinstance(file, list):
            file = [file]
        for fp in file:
            for lemma, word, tags in self.read_file(fp):
                src = [self.source_c2i[BOS]]
                for tag in tags:
                    src.append(self.attr_c2i.get(tag, UNK_IDX))
                for char in lemma:
                    src.append(self.source_c2i.get(char, UNK_IDX))
                src.append(self.source_c2i[EOS])
                trg = [self.target_c2i[BOS]]
                for char in word:
                    trg.append(self.target_c2i.get(char, UNK_IDX))
                trg.append(self.target_c2i[EOS])
                yield src, trg


class LemmaAndTagClassifier(Dataloader):
    def __init__(
            self, train_file, dev_file, test_file=None,
            merge_tags=False, config=None):
        super().__init__()
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file

        self.just_tag = False
        if config:
            self.add_pos, self.tag_in_src, self.unique_tag, \
                self.predict_pos, self.split_tag = \
                    config.add_pos, config.tag_in_src, config.unique_tag, \
                    config.predict_pos, config.split_tag
        else:
            self.add_pos, self.tag_in_src, self.unique_tag, \
                self.predict_pos, self.split_tag =\
                    False, False, True, False, False

        self.train_reader = UnimorphReader(train_file, merge_tags=merge_tags)
        self.dev_reader = UnimorphReader(dev_file, merge_tags=merge_tags)
        self.test_reader = None if not test_file else \
            UnimorphReader(test_file, merge_tags=merge_tags)

        self.batch_data = dict()
        self.nb_train, self.nb_dev, self.nb_test = 0, 0, 0
        self.nb_attr = 0
        self.source, self.target = self.build_vocab()

        self.source_vocab_size = len(self. source)
        self.target_vocab_size = len(self.target)
        self.target_c2i = {c: i for i, c in enumerate(self.target)}
        self.attr_c2i = None
        self.source_c2i = {c: i for i, c in enumerate(self.source)}


    def build_vocab(self):
        char_set, tag_set = set(), set()
        cnts = []

        cnt = 0
        for word, (lemma, tag) in \
                self.train_reader.get_form2lemma_tag(unique=False):
            cnt += 1
            char_set.update(lemma)
            char_set.update(word)
            if self.split_tag:
                tag_set.update(tag.split(";"))
            else:
                tag_set.add(tag)
            if self.add_pos or self.predict_pos:
                pos = get_pos(tag)
                char_set.add(pos)
        cnts.append(cnt)

        self.nb_train = len(self.train_reader.get_form2lemma_tag(
            unique=self.unique_tag))
        self.nb_dev = sum([1 for _ in self.dev_reader.get_form2lemma_tag(
            unique=self.unique_tag)])
        if self.test_reader is None:
            self.nb_test = 0
        else:
            self.nb_test = sum([1 for _ in self.test_reader.get_form2lemma_tag(
                unique=self.unique_tag)])
        chars = sorted(list(char_set))
        tags = sorted(list(tag_set))

        if self.tag_in_src:
            source = [PAD, BOS, EOS, UNK, EOF] + chars + tags
            target = [PAD, BOS, EOS, UNK, EOF] + chars
        else:
            source = [PAD, BOS, EOS, UNK, EOF] + chars
            target = [PAD, BOS, EOS, UNK, EOF] + chars  + tags
        return source, target


    def train_batch_sample(self, batch_size):
        yield from self._batch_sample(
            batch_size, self.train_reader, unique=self.unique_tag)

    def dev_batch_sample(self, batch_size):
        yield from self._batch_sample(
            batch_size, self.dev_reader, unique=self.unique_tag)

    def test_batch_sample(self, batch_size):
        yield from self._batch_sample(
            batch_size, self.test_reader, unique=self.unique_tag)

    def encode_source(self, sent):
        if sent[0] != BOS:
            sent = [BOS] + sent
        if sent[-1] != EOS:
            sent = sent + [EOS]
        l = len(sent)
        s = []
        for x in sent:
            s.append(self.source_c2i[x])
        return torch.tensor(s, device=self.device).view(l, 1)

    def decode_source(self, sent):
        if isinstance(sent, torch.Tensor):
            assert sent.size(1) == 1
            sent = sent.view(-1)
        return [self.source[x] for x in sent]

    def decode_target(self, sent):
        if isinstance(sent, torch.Tensor):
            assert sent.size(1) == 1
            sent = sent.view(-1)
        return [self.target[x] for x in sent]


    def _iter_helper(self, reader, unique):
        for word, (lemma, tag) in reader.get_form2lemma_tag(unique=unique):
            src = [self.source_c2i[BOS]]

            if self.add_pos:
                src.append(self.source_c2i.get(get_pos(tag), UNK_IDX))
            for char in word:
                src.append(self.source_c2i.get(char, UNK_IDX))
            if self.tag_in_src:
                src.append(self.source_c2i[EOF])
                if self.split_tag:
                    for feat in tag.split(";"):
                        trg.append(self.target_c2i.get(feat, UNK_IDX))
                else:
                    src.append(self.source_c2i.get(tag, UNK_IDX))
            src.append(self.source_c2i[EOS])

            trg = [self.target_c2i[BOS]]

            if self.predict_pos:
                trg.append(self.target_c2i.get(get_pos(tag), UNK_IDX))

            # tag first
            if not self.tag_in_src:
                if self.split_tag:
                    for feat in tag.split(";"):
                        trg.append(self.target_c2i.get(feat, UNK_IDX))
                else:
                    trg.append(self.target_c2i.get(tag, UNK_IDX))
                trg.append(self.target_c2i[EOF])

            if not self.just_tag:
                for char in lemma:
                    trg.append(self.target_c2i.get(char, UNK_IDX))

            trg.append(self.target_c2i[EOS])

            yield src, trg, []

    def _batch_helper(self, lst):
        bs = len(lst)
        srcs, trgs, attrs = [], [], []
        max_src_len, max_trg_len, max_nb_attr = 0, 0, 1
        for _, src, trg, attr in lst:
            max_src_len = max(len(src), max_src_len)
            max_trg_len = max(len(trg), max_trg_len)
            srcs.append(src)
            trgs.append(trg)

        batch_src = torch.zeros(
            (max_src_len, bs), dtype=torch.long, device=self.device)
        batch_src_mask = torch.zeros(
            (max_src_len, bs), dtype=torch.float, device=self.device)
        batch_trg = torch.zeros(
            (max_trg_len, bs), dtype=torch.long, device=self.device)
        batch_trg_mask = torch.zeros(
            (max_trg_len, bs), dtype=torch.float, device=self.device)
        for i in range(bs):
            for j in range(len(srcs[i])):
                batch_src[j, i] = srcs[i][j]
                batch_src_mask[j, i] = 1
            for j in range(len(trgs[i])):
                batch_trg[j, i] = trgs[i][j]
                batch_trg_mask[j, i] = 1
        return (batch_src, batch_src_mask, batch_trg,
                batch_trg_mask)

    def _batch_sample(self, batch_size, reader, unique):
        if reader not in self.batch_data:
            lst = list()
            for src, trg, attr in self._iter_helper(reader, unique):
                lst.append((len(src), src, trg, attr))
            self.batch_data[reader] = sorted(lst, key=lambda x: x[0])

        lst = self.batch_data[reader]

        for start in range(0, len(lst), batch_size):
            yield self._batch_helper(lst[start:start + batch_size])

    def train_sample(self):
        for src, trg, tags in self._iter_helper(
                self.train_reader, unique=self.unique_tag):
            yield (
            (torch.tensor(src, device=self.device).view(len(src), 1),
            torch.tensor(trg, device=self.device).view(len(trg), 1)))

    def dev_sample(self):
        for src, trg, tags in self._iter_helper(self.dev_reader, unique=False):
            yield (
            (torch.tensor(src, device=self.device).view(len(src), 1),
            torch.tensor(trg, device=self.device).view(len(trg), 1)))

    def test_sample(self):
        for src, trg, tags in self._iter_helper(self.test_reader, unique=False):
            yield (
            (torch.tensor(src, device=self.device).view(len(src), 1),
            torch.tensor(trg, device=self.device).view(len(trg), 1)))


class SIGMORPHON2019Task2(SIGMORPHON2019Task1):
    def read_file(self, file):
        with open(file, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                toks = line.strip().split('\t')
                if len(toks) < 2 or line[0] == '#':
                    continue
                word, lemma, tags = toks[1], toks[2], toks[5]
                yield list(word), list(lemma), tags.split(';')


class TagSIGMORPHON2019Task1(SIGMORPHON2019Task1):
    def _iter_helper(self, file):
        tag_shift = len(self.source) - self.nb_attr
        if not isinstance(file, list):
            file = [file]
        for fp in file:
            for lemma, word, tags in self.read_file(fp):
                src = []
                src.append(self.source_c2i[BOS])
                for char in lemma:
                    src.append(self.source_c2i.get(char, UNK_IDX))
                src.append(self.source_c2i[EOS])
                trg = []
                trg.append(self.target_c2i[BOS])
                for char in word:
                    trg.append(self.target_c2i.get(char, UNK_IDX))
                trg.append(self.target_c2i[EOS])
                attr = [0] * (self.nb_attr + 1)
                for tag in tags:
                    if tag in self.attr_c2i:
                        attr_idx = self.attr_c2i[tag] - tag_shift
                    else:
                        attr_idx = -1
                    if attr[attr_idx] == 0:
                        attr[attr_idx] = self.attr_c2i.get(tag, 0)
                yield src, trg, attr

    def _batch_helper(self, lst):
        bs = len(lst)
        srcs, trgs, attrs = [], [], []
        max_src_len, max_trg_len, max_nb_attr = 0, 0, 0
        for _, src, trg, attr in lst:
            max_src_len = max(len(src), max_src_len)
            max_trg_len = max(len(trg), max_trg_len)
            max_nb_attr = max(len(attr), max_nb_attr)
            srcs.append(src)
            trgs.append(trg)
            attrs.append(attr)
        batch_attr = torch.zeros(
            (bs, max_nb_attr), dtype=torch.long, device=self.device)
        batch_src = torch.zeros(
            (max_src_len, bs), dtype=torch.long, device=self.device)
        batch_src_mask = torch.zeros(
            (max_src_len, bs), dtype=torch.float, device=self.device)
        batch_trg = torch.zeros(
            (max_trg_len, bs), dtype=torch.long, device=self.device)
        batch_trg_mask = torch.zeros(
            (max_trg_len, bs), dtype=torch.float, device=self.device)
        for i in range(bs):
            for j in range(len(attrs[i])):
                batch_attr[i, j] = attrs[i][j]
            for j in range(len(srcs[i])):
                batch_src[j, i] = srcs[i][j]
                batch_src_mask[j, i] = 1
            for j in range(len(trgs[i])):
                batch_trg[j, i] = trgs[i][j]
                batch_trg_mask[j, i] = 1
        return ((batch_src, batch_attr), batch_src_mask, batch_trg,
                batch_trg_mask)

    def _batch_sample(self, batch_size, file):
        if isinstance(file, list):
            key = tuple(sorted(file))
        else:
            key = file
        if key not in self.batch_data:
            lst = list()
            for src, trg, attr in self._iter_helper(file):
                lst.append((len(src), src, trg, attr))
            self.batch_data[key] = sorted(lst, key=lambda x: x[0])

        lst = self.batch_data[key]
        for start in range(0, len(lst), batch_size):
            yield self._batch_helper(lst[start:start + batch_size])

    def train_sample(self):
        for src, trg, tags in self._iter_helper(self.train_file):
            yield ((torch.tensor(src, device=self.device).view(len(src), 1),
                    torch.tensor(tags, device=self.device).view(1, len(tags))),
                   torch.tensor(trg, device=self.device).view(len(trg), 1))

    def dev_sample(self):
        for src, trg, tags in self._iter_helper(self.dev_file):
            yield ((torch.tensor(src, device=self.device).view(len(src), 1),
                    torch.tensor(tags, device=self.device).view(1, len(tags))),
                   torch.tensor(trg, device=self.device).view(len(trg), 1))

    def test_sample(self):
        for src, trg, tags in self._iter_helper(self.test_file):
            yield ((torch.tensor(src, device=self.device).view(len(src), 1),
                    torch.tensor(tags, device=self.device).view(1, len(tags))),
                   torch.tensor(trg, device=self.device).view(len(trg), 1))


class TagSIGMORPHON2019Task2(TagSIGMORPHON2019Task1, SIGMORPHON2019Task2):
    pass
