from __future__ import annotations

import typing
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import unidecode
from scipy.ndimage.filters import median_filter


class Image(object):
    orb = None

    def __init__(self, image_path: typing.Optional[Path] = None, image_data: typing.Optional[np.ndarray] = None):
        self.image_path = image_path
        if image_data is not None:
            self.image_data: np.ndarray = image_data
        elif image_path is not None:
            self.image_data: np.ndarray = cv2.imread(str(image_path))
            if self.image_data is None:
                raise Exception("Image doesn't exist: {}".format(image_path))
        else:
            raise Exception("No Image source specified")
        self._original_image: np.ndarray = self.image_data.copy()

        self.detector = None
        self.keypoints = None
        self.descriptor = None
        self.text = None
        self.bag = None

    def export(self, file_path: Path):
        cv2.imwrite(str(file_path), self.image_data)

    def save(self):
        self._original_image = self.image_data

    def reset(self):
        # self.keypoints = None
        # self.descriptor = None
        # self.text = None
        # self.bag = None
        self.image_data = self._original_image.copy()

    def grayscale(self):
        self.image_data = cv2.cvtColor(self.image_data, cv2.COLOR_RGB2GRAY)
        return self

    def show(self, window_title="image"):
        cv2.imshow(window_title, self.image_data)
        cv2.waitKey()

    # <-127,127> for both contrast and brightness
    def set_contrast_brightness(self, contrast: int, brightness: int):
        img = np.int16(self.image_data)
        img = img * (contrast / 127 + 1) - contrast + brightness
        img = np.clip(img, 0, 255)
        self.image_data = np.uint8(img)

    def resize(self, w: int, h: int):
        self.image_data = cv2.resize(
            self.image_data, (w, h), interpolation=cv2.INTER_CUBIC)

    def rescale(self, fx=3, fy=3):
        self.image_data = cv2.resize(
            self.image_data, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

    def remove_noise(self):
        # Apply dilation and erosion to remove some noise
        kernel = np.ones((3, 3), np.uint8)
        self.image_data = cv2.dilate(self.image_data, kernel, iterations=1)
        self.image_data = cv2.erode(self.image_data, kernel, iterations=1)

        # Apply blur to smooth out the edges
        self.image_data = cv2.GaussianBlur(self.image_data, (5, 5), 0)

    def normalize(self):
        self.image_data = cv2.normalize(
            self.image_data,
            self.image_data,
            0,
            255,
            cv2.NORM_MINMAX
        )

    def binarize(self):
        # ret, self.image_data = cv2.threshold(
        #     self.image_data, 200, 255, cv2.THRESH_BINARY)
        self.image_data = cv2.adaptiveThreshold(
            self.image_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    def sharpen(self):
        self.image_data = unsharp_mask(
            self.image_data, threshold=0.1, amount=4.7, sigma=3.8)

    def ocr_preprocessing(self):
        # self.set_contrast_brightness(125, -100)
        self.remove_noise()
        self.rescale(2, 2)
        self.grayscale()

        # self.sharpen()
        self.image_data = cv2.erode(
            self.image_data, np.ones((1, 1)), iterations=-1)

        self.normalize()
        self.binarize()

    def crop_by_bounding_box(self, box: typing.Tuple[int, int, int, int]) -> Image:
        left, right, top, bottom = box
        return Image(self.image_path + ".cropped", self.image_data[top:bottom, left:right])

    def get_keypoints_and_descriptors(self):
        """
            Computes, sets instance variables and returns keypoints and descriptor using ORB.
        """

        if self.keypoints is not None and self.descriptor is not None:
            return self.keypoints, self.descriptor

        if self.detector is None:
            self.detector = cv2.ORB_create()

        self.reset()
        self.keypoints, self.descriptor = self.detector.detectAndCompute(
            self.image_data, None
        )

        return self.keypoints, self.descriptor

    def get_text(self, language: typing.Optional[str] = None) -> str:
        if self.text is not None:
            return self.text

        self.ocr_preprocessing()
        try:
            self.text = pytesseract.image_to_string(
                self.image_data, lang=language)
        except Exception as e:
            print("ERROR DETECTING TEXT: ", e, file=sys.stderr)
            self.text = ""
        self.reset()

        return self.text

    def get_bag_of_words(self, language: typing.Optional[str] = None) -> typing.Dict[str, int]:
        if self.bag is not None:
            return self.bag

        raw_text = self.get_text(language)

        bag = {}
        arr = re.findall("\w{3,}", raw_text)
        for word in arr:
            # word = unidecode.unidecode(word)
            if word in bag:
                bag[word] += 1
            else:
                bag[word] = 1

        self.bag = bag

        return self.bag


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
