from pathlib import Path

import cv2
import numpy as np
import pdf2image
import tempfile

from .slide import Slide


class Presentation():
    def __init__(self, pdf_file_path: Path):
        print("[I] Initializing presentation from PDF file: ", pdf_file_path)

        self.file_path = pdf_file_path
        with open(pdf_file_path, "rb") as pdf_file:
            pdf_raw = pdf_file.read()
        with tempfile.TemporaryDirectory() as path:
            pages = pdf2image.convert_from_bytes(pdf_raw, output_folder=path)
            self.pages = []
            for i, page in enumerate(pages):
                slide_image = Slide(np.array(page), self, i)
                slide_image.image_data = cv2.cvtColor(
                    slide_image.image_data, cv2.COLOR_RGB2BGR)
                slide_image.save()
                self.pages.append(slide_image)

    def export(self, output_dir: Path, file_prefix: str = "slide", extension: str = "jpg", new_width: int = -1):
        for i, page in enumerate(self.pages):
            if new_width > 0:
                height, width, depth = page.image_data.shape
                new_height = int(new_width * height / width)
                page.resize(new_width, new_height)
            page.export(
                (output_dir / "{}-{:03d}.{}".format(file_prefix, i, extension)))
            page.reset()
