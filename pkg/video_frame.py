import typing

import PIL
import imagehash

from .image import Image


class Screen():
    def __init__(self, bndbox: typing.Tuple[int, int, int, int], frame: Image):
        self.bndbox = bndbox
        self.image = frame.crop_by_bounding_box(bndbox)

        pilim = PIL.Image.fromarray(self.image.image_data)
        self.hash = imagehash.phash(pilim, hash_size=12)

        self.match = None
        self.skipped = False

    def add_match(self, match):
        self.match = match


class VideoFrame(Image):
    def __init__(self, raw_image, video, frame_no):
        Image.__init__(
            self,
            "frame://{}#{}".format(video.video_path, frame_no),
            raw_image
        )

        self.video = video
        self.frame_no = int(frame_no)
        self.screens = []

    def add_screen(self, screen: Screen):
        self.screens.append(screen)

    def filter_screens(self):
        screens = []

        for screen in self.screens:
            if screen.skipped:
                continue
            screens.append(screen)

        self.screens = screens
