# Morphologically aware word-level translation

This repository contains the code for experiments described in the following paper:

- Paula Czarnowska, Sebastian Ruder, Ryan Cotterell, and Ann Copestake. 2019. **[Morphologically Aware Word-level Translation](https://aclanthology.org/2020.coling-main.256/)**. In *Proceedings of the 28th International Conference on Computational Linguistics (COLING)*.

To reproduce the results for the baselines, base, hybrid and oracle models on the morphologically rich dictionaries follow the commands from the reproduce.sh script.

## Acknowledgements

#### Translator code
The code in *code* directory is the refactored and extended version of the [latent-variable-vecmap](https://github.com/sebastianruder/latent-variable-vecmap) repository, containing the code for experiments from the followin paper:

- Sebastian Ruder, Ryan Cotterell, Yova Kementchedjhieva, and  Anders Søgaard. 2018. **[A Discriminative Latent-Variable Model for Bilingual Lexicon Induction](https://aclanthology.org/D18-1042/)**. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

The [latent-variable-vecmap](https://github.com/sebastianruder/latent-variable-vecmap) was, in turn, based on the [vecmap](https://github.com/artetxem/vecmap) repository, tied to the papers:

- Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2018. **[A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings](https://aclweb.org/anthology/P18-1073)**. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*.
- Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2018. **[Generalizing and improving bilingual word embedding mappings with a multi-step framework of linear transformations](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16935/16781)**. In *Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18)*, pages 5012-5019.
- Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2017. **[Learning bilingual word embeddings with (almost) no bilingual data](https://aclweb.org/anthology/P17-1042)**. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 451-462.
- Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2016. **[Learning principled bilingual mappings of word embeddings while preserving monolingual invariance](https://aclweb.org/anthology/D16-1250)**. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, pages 2289-2294.

#### Transducer code
The code in *reinflector* directory is an extended version of the [crosslingual-inflection-baseline](https://github.com/sigmorphon/crosslingual-inflection-baseline) repository, which provides code for the SIGMORPHON 2019 baselines:

-  Arya D. McCarthy, Ekaterina Vylomova, Shijie Wu, Chaitanya Malaviya, Lawrence Wolf-Sonkin, Garrett Nicolai, Christo Kirov, Miikka Silfverberg, Sabrina J. Mielke, Jeffrey Heinz, Ryan Cotterell and Mans Hulden. 2019. **[The SIGMORPHON 2019 Shared Task: Morphological Analysis in Context and Cross-Lingual Transfer for Inflection](https://aclanthology.org/W19-4226/)**. In *Proceedings of the 16th Workshop on Computational Research in Phonetics, Phonology, and Morphology*.

