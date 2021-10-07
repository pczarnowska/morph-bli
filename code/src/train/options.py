from enum import Enum
import enum


class InputType(enum.IntEnum):
    SRC = 1
    TRG = 2


class ModelType(enum.IntEnum):
    LAT_MORPH = 1
    LAT_VAR = 2
    SELF_LEARN = 3
    NOT_SELF_LEARN = 4


class InputData():
    def __init__(self, type, lang):
        self.type = type
        self.lang = lang
        self.words = None
        self.word2ind = None

        self.oov_words = None

        self.um = None
        self.tags = None
        self.tag_ids = None

        # self.general_tags = None
        # self.gen_tag_ids = None

    def retrieve_word2ind(self):
        self.word2ind = {word: i for i, word in enumerate(self.words)}


class Options(object):
    def __init__(self, args, xp, dtype):
        self.xp = xp
        self.dtype = dtype
        self.encoding = args.encoding
        self.log = args.log
        self.verbose = args.verbose

        self.model_type, self.lat_model = self.get_model_type(args)
        self.match_constr = args.match_constr

        self.src_data = InputData(InputType.SRC, args.src_lang)
        self.trg_data = InputData(InputType.TRG, args.trg_lang)

        self.no_random_matching = args.no_random_matching

        # training details
        self.n_similar = args.n_similar
        self.chunk_size = args.chunk_size
        self.n_repeats = args.n_repeats
        self.rank_constr = args.rank_constr
        self.asym = args.asym
        self.threshold = args.threshold
        self.orthogonal = args.orthogonal
        self.unconstrained = args.unconstrained
        self.whiten = args.whiten
        self.src_reweight, self.trg_reweight =\
            args.src_reweight, args.trg_reweight
        self.src_dewhiten, self.trg_dewhiten =\
            args.src_dewhiten, args.trg_dewhiten
        self.dim_reduction = args.dim_reduction
        self.direction = args.direction

        self.good_inds = None
        self.src_tag_masks = None

        # Maximum dimensions for the similarity matrix computation in memory
        # A MAX_DIM_X * MAX_DIM_Z dimensional matrix will be used
        self.max_dimx = 10000
        self.max_dimz = 10000

        self.morph_feats = self.get_mfeats(args.morph_feats)
        self.single_tag = args.single_tag
        self.tag2ind = None

    def get_mfeats(self, mfeats):
        if not mfeats or mfeats == "-":
            return None
        else:
            return mfeats.split(";")

    def get_model_type(self, args):
        if args.lat_morph:
            return ModelType.LAT_MORPH, True
        elif args.lat_var:
            return ModelType.LAT_VAR, True
        elif args.self_learning:
            return ModelType.SELF_LEARN, False
        else:
            return ModelType.NOT_SELF_LEARN, False

    def get_data(self, input_type):
        if input_type == InputType.SRC:
            return self.src_data
        elif input_type == InputType.TRG:
            return self.trg_data
        else:
            return None
