from pathlib import Path

import cv2

from .image import Image


class Slide(Image):
    def __init__(self, raw_image, presentation, slide_no):
        Image.__init__(
            self,
            "slide://{}#{}".format(presentation.file_path, slide_no),
            raw_image
        )

        self.presentation = presentation
        self.slide_no = slide_no

    def export(self, file_path: Path):
        cv2.imwrite(str(file_path), self.image_data)
        self.file_path = str(file_path)
