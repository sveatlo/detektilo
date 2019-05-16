import math
import typing
from pathlib import Path

import cv2
import numpy as np
import unidecode
from matplotlib import pyplot as plt

from .image import Image
from .misc import deduplicate_list, cosine_similarity, find_better_word


# FLANN_INDEX_LSH = 6


class Matcher():
    """
        Matches a single image against a set of images
    """
    # index_params = dict(algorithm=FLANN_INDEX_LSH,
    #                     table_number=6,  # 12
    #                     key_size=12,     # 20
    #                     multi_probe_level=1)  # 2

    # search_params = dict(checks=50)

    def __init__(self, language: typing.Optional[str] = None):
        print("[I] Initializing Matcher...")
        self.language = language
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self._candidate_images: typing.List[Image] = []

        self._training_bags = []
        self._dictionary = []

    @classmethod
    def from_paths(self, language: typing.Optional[str] = None, *candidate_images_paths: typing.List[Path]):
        return Matcher.from_images(language, *[Image(training_image_path) for training_image_path in candidate_images_paths])

    @classmethod
    def from_images(self, language: typing.Optional[str] = None, *candidate_images: typing.List[Path]):
        matcher = Matcher(language)
        matcher.set_candidate_images(candidate_images)

        return matcher

    def set_candidate_images(self, images: typing.List[Image]):
        self._candidate_images = []
        self._training_bags = []
        self._dictionary = []

        self.matcher.clear()

        for image in images:
            print("\t[D] Adding candidate image: ", image.image_path)
            # print("Shape", image.image_data.shape)
            # plt.imshow(image.image_data)
            # plt.show()

            image.get_bag_of_words(self.language)
            image.get_keypoints_and_descriptors()

            self._candidate_images.append(image)

            self.matcher.add([image.descriptor])

            self._training_bags.append(image.bag)
            self._dictionary.extend(list(image.bag.keys()))

        self.matcher.train()
        self._deduplicate_dictionary()

    def _deduplicate_dictionary(self):
        self._dictionary = deduplicate_list(self._dictionary)

    def match(self, query_image: Image, do_keypoints: bool = True, do_text: bool = True, keypoints_weight: float = 1, text_weight: float = 1):
        assert len(self._candidate_images) > 0, "Matcher has no candidate images"
        assert do_text or do_keypoints, "At least one matching method is required"

        # query_image.reset()
        # query_image.get_keypoints_and_descriptors()
        # query_image.ocr_preprocessing()

        if do_keypoints:
            keypoints_score, matches = self._match_keypoints(query_image)
        if do_text:
            text_score = self._match_text(query_image)

        scores = [0] * len(self._candidate_images)
        for i in range(len(self._candidate_images)):
            # scores[i] = (keypoints_weight * keypoints_score[i] + text_weight * text_score[i]) / (keypoints_weight + text_weight)
            scores[i] = 0
            total_weight = 0
            if do_keypoints:
                scores[i] += keypoints_score[i] * keypoints_weight
                total_weight += keypoints_weight
            if do_text:
                scores[i] += text_score[i] * text_weight
                total_weight += text_weight

            scores[i] /= total_weight

        if not do_keypoints:
            matches = []

        return Match(query_image, self._candidate_images, scores, matches)

    def _match_keypoints(self, query_image: Image) -> typing.Tuple[typing.List[int], typing.List[typing.Any]]:
        """
            Match query image to candidate images using keypoints.
            Returns array of scores per image
        """
        query_image.get_keypoints_and_descriptors()

        # matches = self.matcher.match(query_image.descriptor)
        matches = self.matcher.knnMatch(query_image.descriptor, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 1:
                good.append(m_n[0])
                continue
            elif len(m_n) != 2:
                continue
            (m, n) = m_n
            if m.distance < 0.7 * n.distance:
                good.append(m)

        images_scores = [0] * len(self._candidate_images)
        images_matches = [None] * len(self._candidate_images)
        for image_index, image in enumerate(self._candidate_images):
            matches_scores = []
            matches = []
            for i, match in enumerate(good):
                if match.imgIdx != image_index:
                    continue
                matches.append(match)
                matches_scores.append((256 - match.distance) / 256)

            match_cnt = len(matches_scores)
            if match_cnt <= 0:
                continue

            images_scores[image_index] = (
                0.5 + ((math.tanh(match_cnt / 3 - 1)) / 2)) * (sum(matches_scores) / match_cnt)

            images_matches[image_index] = matches

        return images_scores, images_matches

    def _match_text(self, query_image: Image):
        original_query_bag = query_image.get_bag_of_words(self.language)
        query_bag = {}
        for word, count in original_query_bag.items():
            query_bag[find_better_word(word, self._dictionary)] = count

        dictionary = self._dictionary.copy()
        dictionary.extend(list(query_bag))
        dictionary = deduplicate_list(dictionary)

        bags = [query_bag]
        bags.extend(self._training_bags.copy())

        vectors = []
        for bag in bags:
            vector = [0] * len(dictionary)
            for i, word in enumerate(dictionary):
                if word in bag:
                    vector[i] = bag[word]

            vectors.append(vector)

        query_vector = vectors[0]
        train_vectors = vectors[1:]

        return [cosine_similarity(query_vector, train_vector) for train_vector in train_vectors]


class Match():
    def __init__(self, query_image: Image, images: typing.List[Image], scores: typing.List[float], kp_matches: typing.List[typing.Any]):
        if len(kp_matches) <= 0:
            kp_matches = [None] * len(images)
        self.query_image = query_image

        self._best_index: int = -1
        self.candidates = list(zip(images, scores, kp_matches))

        if len(self.candidates) <= 0:
            return

        self.candidates.sort(key=lambda c: c[1], reverse=True)
        if self.candidates[0][1] != 0:
            self._best_index = 0
        self.candidates = self.candidates[:3]
        # first_zero = self.candidates.index(
        # next(filter(lambda c: c != 0, self.candidates)))
        # self.candidates = self.candidates[:first_zero]

    def best_index(self):
        return self._best_index

    def best_image(self):
        if self._best_index < 0:
            return None

        return self.candidates[self._best_index][0]

    def best_match(self):
        if self._best_index < 0:
            return None

        return self.candidates[self._best_index]

    def visualize_best(self):
        if self._best_index < 0:
            return np.zeros((600, 400))

        draw_params = dict(
            matchColor=(0, 255, 0),  # draw matches in green color
            singlePointColor=(255, 0, 0),
            # matchesMask = matchesMask,  # draw only inliers
            flags=2
        )

        kp_matches = self.candidates[self._best_index][2]

        return cv2.drawMatches(self.query_image.image_data, self.query_image.keypoints,
                               self.best_image().image_data, self.best_image().keypoints, kp_matches, None, **draw_params)

    def plot_best(self, window_name="Best match"):
        plt.imshow(self.visualize_best())
        plt.show()
