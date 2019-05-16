import math
import typing

import cv2
import numpy as np


def deduplicate_list(mylist):
    return list(dict.fromkeys(mylist))


def cosine_similarity(vectorA, vectorB) -> float:
    dot_AB = np.dot(vectorA, vectorB)
    magn_A = math.sqrt(sum([math.pow(Ai, 2) for Ai in vectorA]))
    magn_B = math.sqrt(sum([math.pow(Bi, 2) for Bi in vectorB]))

    if magn_A == 0 or magn_B == 0:
        return 0

    return dot_AB / (magn_A * magn_B)


def match_width(imgs, m=True):
    fn = min
    if not m:
        fn = max
    min_width = fn(img.shape[1] for img in imgs if img is not None)
    for i, img in enumerate(imgs):
        height, width = img.shape[0], img.shape[1]
        new_height = int(min_width * height / width)
        imgs[i] = cv2.resize(img, (min_width, new_height),
                             interpolation=cv2.INTER_CUBIC)

    return imgs


def match_height(imgs, m=True):
    fn = min
    if not m:
        fn = max
    min_height = min(img.shape[0] for img in imgs if img is not None)
    for i, img in enumerate(imgs):
        height, width = img.shape[0], img.shape[1]
        new_width = int(min_height * width / height)
        imgs[i] = cv2.resize(img, (new_width, min_height),
                             interpolation=cv2.INTER_CUBIC)

    return imgs


def min_edit_dist(word1, word2):
    """
    Find minimal distance between words
    OP: https://stackoverflow.com/a/20092392
    """
    len_1 = len(word1)
    len_2 = len(word2)
    # the matrix whose last element ->edit distance
    x = [[0] * (len_2 + 1) for _ in range(len_1 + 1)]
    for i in range(0, len_1 + 1):
        # initialization of base case values
        x[i][0] = i
        for j in range(0, len_2 + 1):
            x[0][j] = j
    for i in range(1, len_1 + 1):
        for j in range(1, len_2 + 1):
            if word1[i - 1] == word2[j - 1]:
                x[i][j] = x[i - 1][j - 1]
            else:
                x[i][j] = min(x[i][j - 1], x[i - 1][j], x[i - 1][j - 1]) + 1
    return x[i][j]


def find_better_word(word: str, dictionary: typing.List) -> str:
    distances = [min_edit_dist(word, dict_word) for dict_word in dictionary]
    min_index = np.argmin(distances)
    if distances[min_index] > 2:
        return word
    return dictionary[min_index]
