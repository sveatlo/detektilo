import cv2
import imagehash
import xml.etree.ElementTree
import sys

from PIL import Image
from pathlib import Path

hashes = {}


class Extractor():
    def __init__(self, root_path, video_file_path, images_dir, skipped_frames=1000, interactive=False):
        self.video_file = str(video_file_path)

        event_name = list(video_file_path.parts)[
            len(list(root_path.parts)) - 1:-1]
        self.images_path = Path(
            "{}/{}".format(images_dir, "-".join(event_name)))

        self.skipped_frames = skipped_frames
        self.cap = None
        self.frames_cnt = 0
        self.interactive = interactive

    def process(self):
        self.cap = cv2.VideoCapture(self.video_file)
        self.frames_cnt = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.images_path.mkdir(parents=True, exist_ok=True)

        i = 0
        while self.cap.isOpened():
            # get image
            ok, frame = self.cap.read()
            if not ok:
                break

            # skip
            i += self.skipped_frames
            if i > self.frames_cnt:
                i = self.frames_cnt
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)

            # save the image
            image = Image.fromarray(frame)
            hash = str(imagehash.phash(image, hash_size=12))
            if hash in hashes:
                continue
            hashes[hash] = True

            # show image
            if self.interactive:
                cv2.imshow('frame', frame)
                b = False
                r = False
                while True:
                    k = cv2.waitKey(0)
                    if k & 0xFF == ord('q'):  # quit
                        b = True
                        break
                    elif k & 0xFF == ord('r'):  # reject
                        r = True
                        break
                    elif k & 0xFF == ord('a'):  # accept
                        break
                if b:
                    break
                elif r:
                    continue  # skip to next frame

            # save image
            image_path = "{}/{}.jpg".format(str(self.images_path), i)
            if not Path(image_path).exists():
                cv2.imwrite(image_path, frame)
        try:
            self.images_path.rmdir()
        except Exception as e:
            pass
