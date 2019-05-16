import typing
from pathlib import Path

import cv2
import numpy as np

from .video_frame import VideoFrame


class Video():
    def __init__(self, video_path: Path):
        self.video_path: Path = video_path
        self.capture: cv2.VideoCapture = cv2.VideoCapture(str(self.video_path))

        self.fps: int = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.frames_cnt: int = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_shape: typing.Tuple[int, int] = (
            self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

    def __del__(self):
        self.capture.release()

    def get_current_frame_no(self) -> float:
        return self.capture.get(cv2.CAP_PROP_POS_FRAMES)

    def _seek_frame(self, frame_no):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    def seek(self, seconds: int):
        self._seek_frame(seconds * self.fps)

    def forward(self, seconds: int):
        self._seek_frame(self.get_current_frame_no() + seconds * self.fps)

    def get_current_frame(self) -> VideoFrame:
        """
        VideoFrame object or None on failure
        """
        ok, raw = self.capture.read()
        if not ok:
            return None

        return VideoFrame(raw, self, self.get_current_frame_no())
