# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from src.utils.cupy_utils import *

def read(file_name, encoding='utf-8', dtype='float', threshold=0, vocabulary=None, xp=None):
    if not xp:
        xp = np

    file = open(file_name, encoding=encoding, errors='surrogateescape')
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    matrix = xp.empty((count, dim), dtype=dtype) if vocabulary is None else []
    for i in range(count):
        splitt = file.readline().split(' ', 1)
        if len(splitt) != 2:
            continue

        #print(splitt)
        word, vec = splitt
        if vocabulary is None:
            words.append(word)
            matrix[i] = xp.array(np.fromstring(vec, sep=' ', dtype=dtype))
        elif word in vocabulary:
            words.append(word)
            matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
    file.close()
    matrix = matrix[:len(words)]
    return (words, matrix) if vocabulary is None else (words, xp.array(matrix, dtype=dtype))

def write(words, matrix, file):
    m = asnumpy(matrix)
    print('%d %d' % m.shape, file=file)
    for i in range(len(words)):
        print(words[i] + ' ' + ' '.join(['%.6g' % x for x in m[i]]), file=file)


def length_normalize(matrix):
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix = np.divide(matrix, norms[:, xp.newaxis])
    return matrix


def mean_center(matrix, return_means=False):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=0)
    matrix = np.subtract(matrix, avg)
    if return_means:
        return matrix, avg
    return matrix


def length_normalize_dimensionwise(matrix, return_norms=False):
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(matrix**2, axis=0))
    norms[norms == 0] = 1
    matrix = np.divide(matrix, norms)
    if return_norms:
        return matrix, norms
    return matrix


def mean_center_embeddingwise(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=1)
    matrix = np.subtract(matrix, avg[:, xp.newaxis])
    return matrix


def normalize(vectors, action_list, return_means_and_norms=False):
    """
    Perform each type of normalisation from the list (in the given order).
    :param embeddings:
    :param action_list:
    :return:
    """
    means = None
    norms = None
    for action in action_list:
        if action == 'unit':
            vectors = length_normalize(vectors)
        elif action == 'center':
            vectors, means = mean_center(vectors, return_means=True)
        elif action == 'unitdim':
            vectors, norms = length_normalize_dimensionwise(vectors, return_norms=True)
        elif action == 'centeremb':
            vectors = mean_center_embeddingwise(vectors)
    if return_means_and_norms:
        return vectors, means, norms
    return vectors


def normalize_oovs(vectors, action_list, means, norms):
    for action in action_list:
        if action == 'unit':
            vectors = length_normalize(vectors)
        elif action == 'center':
            vectors = vectors - means
        elif action == 'unitdim':
            vectors = vectors/norms
        elif action == 'centeremb':
            vectors = mean_center_embeddingwise(vectors)
    return vectors


def normalize_oovs_based_on_m(vectors, action_list, m):
    """
    Perform each type of normalisation from the list (in the given order).
    The dimensionwise normalisation is performed based on means/norms
    obtained for the matrix of non-oov embeddings (m).
    :param embeddings:
    :param action_list:
    :return:
    """
    for action in action_list:
        if action == 'unit':
            vectors = length_normalize(vectors)
        elif action == 'center':
            _, means = mean_center(m, return_means=True)
            vectors = vectors - means
        elif action == 'unitdim':
            _, norms = length_normalize_dimensionwise(m, return_norms=True)
            vectors = vectors/norms
        elif action == 'centeremb':
            vectors = mean_center_embeddingwise(vectors)
    return vectors