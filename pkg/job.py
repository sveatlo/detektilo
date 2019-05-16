import random
import string
import typing
from pathlib import Path

from lxml import etree

from .matcher import Match
from .presentation import Presentation
from .slide import Slide
from .video import Video
from .video_frame import VideoFrame


class Job():
    def __init__(self, presentation: Presentation, video: Video, output_folder: Path, job_name="video"):
        self.job_name = job_name
        self.output_folder = output_folder

        (self.output_folder / "detected").mkdir(parents=True, exist_ok=True)
        (self.output_folder / "matched").mkdir(parents=True, exist_ok=True)
        (self.output_folder / "slides/jpg_240").mkdir(parents=True, exist_ok=True)
        (self.output_folder / "slides/jpg_640").mkdir(parents=True, exist_ok=True)
        (self.output_folder / "slides/jpg_1024").mkdir(parents=True, exist_ok=True)

        self.presentation = presentation
        self.video = video

        # export slides
        self.presentation.export(
            self.output_folder / "slides/jpg_240", new_width=240)
        self.presentation.export(
            self.output_folder / "slides/jpg_640", new_width=640)
        self.presentation.export(
            self.output_folder / "slides/jpg_1024", new_width=1024)

        self.processed_frames: typing.List[VideoFrame] = []

    def add_processed_frame(self, frame: VideoFrame):
        self.processed_frames.append(frame)

    def get_last_processed_frame(self) -> VideoFrame:
        if len(self.processed_frames) > 0:
            return self.processed_frames[-1]

        return None

    def get_last_screen_hashes(self):
        frame = self.get_last_processed_frame()
        if frame is None:
            return []

        return [screen.hash for screen in frame.screens]

    def filter_frames(self):
        new_frames = []

        slides_in_prev_frame = ""
        for frame in self.processed_frames:
            frame.filter_screens()
            if len(frame.screens) <= 0:
                continue

            slides_in_frame = []
            for screen in frame.screens:
                try:
                    slides_in_frame.append(
                        screen.match.best_match()[0].image_path)
                except Exception as e:
                    print(
                        f"[W] Failed to get besties from previous slide: {e}")
                    pass

            slides_in_frame.sort()
            slides_in_frame = "".join(slides_in_frame)
            if slides_in_frame == slides_in_prev_frame:
                slides_in_prev_frame = slides_in_frame
                continue
            slides_in_prev_frame = slides_in_frame

            new_frames.append(frame)

        self.processed_frames = new_frames

    def output(self, output_compatible: bool = False):
        self.filter_frames()

        print(f"[I] Processing output... Compatible = {output_compatible}")
        if output_compatible:
            xml_root = self._output_compatible()
        else:
            xml_root = self._output_new()

        job_xml_string = etree.tostring(xml_root, pretty_print=True)
        with open(str(self.output_folder / "slides.xml"), "wb") as output_file:
            output_file.write(job_xml_string)

    def _output_compatible(self):
        job_el = etree.Element("job", name=self.job_name, video_src=self.video.video_path,
                               presentation_src=self.presentation.file_path)

        # sources
        sources_el = etree.Element("sources")
        video_el = etree.Element("video", src=str(self.video.video_path))
        presentation_el = etree.Element(
            "presentation", src=str(self.presentation.file_path))
        sources_el.append(video_el)
        sources_el.append(presentation_el)
        job_el.append(sources_el)

        # slides
        slides_el = etree.Element("slides")
        for i, frame in enumerate(self.processed_frames):
            all_frame_matches = []
            all_screens = []

            for screen in frame.screens:
                bestie = screen.match.best_match()
                if bestie is None:
                    continue
                all_frame_matches.append(bestie)
                all_screens.append(screen)

            screen_match_tuples = list(zip(all_screens, all_frame_matches))

            next_frame_no = -1
            if len(self.processed_frames) > i + 1:
                next_frame_no = self.processed_frames[i + 1].frame_no
            else:
                next_frame_no = self.video.frames_cnt

            slide_el = etree.Element("slide")

            screen_match_tuples.sort(
                key=lambda c: c[1][1], reverse=True)
            if len(screen_match_tuples) <= 0:
                continue
            bestie = screen_match_tuples[0]
            if bestie is None:
                continue

            # save screen image
            screen_img_path = self.output_folder / \
                ("detected/screen-{:08d}.jpg".format(frame.frame_no, i))
            bestie[0].image.export(screen_img_path)
            # save best match
            bestie_img_path = self.output_folder / \
                ("matched/{}.jpg".format(''.join(random.choice(string.ascii_lowercase)
                                                 for i in range(8))))
            bestie[1][0].export(bestie_img_path)

            slide_el.set("start", str(int(frame.frame_no / frame.video.fps)))
            slide_el.set("end", str(int(next_frame_no / frame.video.fps) - 1))
            slide_el.set("no", str(bestie[1][0].slide_no))
            slide_el.set("src", str(screen_img_path))
            slide_el.set("match_src", str(bestie_img_path))
            slides_el.append(slide_el)

        job_el.append(slides_el)

        return job_el

    def _output_new(self):
        job_el = etree.Element("job", name=self.job_name, video_src=self.video.video_path,
                               presentation_src=self.presentation.file_path)

        for frame in self.processed_frames:
            frame_el = etree.Element("frame",
                                     no=str(frame.frame_no),
                                     time=str(
                                         int(frame.frame_no / frame.video.fps))
                                     )

            for i, screen in enumerate(frame.screens):
                # save screen image
                screen_img_path = self.output_folder / \
                    ("detected/screen-{:08d}-{:02d}.jpg".format(frame.frame_no, i))
                screen.image.export(screen_img_path)

                # generate screen xml tree
                screen_el = etree.Element("screen")

                # add cropped screen img
                screen_source_el = etree.Element("img")
                screen_source_el.text = str(screen_img_path)
                screen_el.append(screen_source_el)

                # add screen bndbox
                bndbox_el = etree.Element("bndbox")
                xmin = etree.Element("xmin")
                xmax = etree.Element("xmax")
                ymin = etree.Element("ymin")
                ymax = etree.Element("ymax")
                xmin.text = str(screen.bndbox[0])
                xmax.text = str(screen.bndbox[1])
                ymin.text = str(screen.bndbox[2])
                ymax.text = str(screen.bndbox[3])
                bndbox_el.append(xmin)
                bndbox_el.append(xmax)
                bndbox_el.append(ymin)
                bndbox_el.append(ymax)
                screen_el.append(bndbox_el)

                # add best match
                if screen.match is not None:
                    screen_el.append(self._create_match_element(
                        screen.match.best_match()))

                    # add candidates
                    candidates_el = etree.Element("candidates")
                    for candidate in screen.match.candidates:
                        candidates_el.append(
                            self._create_match_element(candidate))
                    screen_el.append(candidates_el)

                frame_el.append(screen_el)

            job_el.append(frame_el)

        return job_el

    def _create_match_element(self, match) -> etree.Element:
        match_el = etree.Element("match")

        if match is None:
            return match_el

        score_el = etree.Element("score")
        score_el.text = str(match[1])
        match_el.append(score_el)

        screen_img_path = self.output_folder / \
            ("matched/{}.jpg".format(''.join(random.choice(string.ascii_lowercase)
                                             for i in range(8))))
        match[0].export(screen_img_path)

        screen_source_el = etree.Element("img")
        screen_source_el.text = str(screen_img_path)
        match_el.append(screen_source_el)

        if isinstance(match[0], Slide):
            slide_el = etree.Element("slide")
            slide_el.text = str(match[0].slide_no)
            match_el.append(slide_el)

        return match_el
