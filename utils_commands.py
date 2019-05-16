import io
import math
import random
import string
import sys
import time
import typing

import PIL
import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree
from pathlib import Path
from object_detection.utils import dataset_util

from pkg import Detector, Extractor, Image, Matcher, Presentation, Video, match_height


@click.group("utils")
def utils_group():
    pass


@utils_group.command(name="extract-screenshots", help="Used for generating the dataset of screenshots from many videos organizing them into 1-level hierarchy based on the video.")
@click.option("--images-dir", "-id", "images_dir", required=True, help="Path to saved screenshots.")
@click.option("--video-dir", "-vd", "video_dir", required=True, help="Path to videos dir. Will be searched recursively for <filename>")
@click.option("--skipped-frames", "-s", "skipped_frames", default=1000, help="Number of frames to skip before taking another screenshot", show_default=True)
@click.option("--video-filename", "-f", "video_filename", default="video_HD.webm", show_default=True)
@click.option("--interactive", "-i", "interactive", type=bool, default=False, is_flag=True, help="Run interactively - use [a] to accept, [r] to reject, [q] to quit")
def extract_screenshots(images_dir, video_dir, video_filename, skipped_frames, interactive):
    root_path = Path(video_dir)
    videos = []
    for video_path in root_path.glob("**/" + video_filename):
        e = Extractor(root_path, video_path, images_dir,
                      skipped_frames, interactive)
        videos.append(e)

    with click.progressbar(videos, width=0, show_eta=False) as bar:
        for extractor in bar:
            extractor.process()


@utils_group.command(name="generate-tfrecords", help="Create TFRecord file for training and evaluation")
@click.option("--images-dir", "-i", "images_dir", required=True, help="Path to saved screenshots.")
@click.option("--class-mapping", "-cm", "class_mapping", type=(str, int), multiple=True, default=("screen", 1))
@click.option("--train-name", "train_file_name", type=str, default="train.record", help="Name of the training TFRecord file")
@click.option("--test-name", "test_file_name", type=str, default="test.record", help="Name of the evaluation TFRecord file")
@click.option("--output-dir", "-o", "output_dir", type=click.Path(exists=True), required=True, help="Where to output the TFRecord files")
@click.option("--eval-percentage", "-p", "eval_percentage", default=30, type=click.IntRange(0, 50), help="Percentage of all images to be put into eval set.", show_default=True)
def generate_tfrecords(images_dir, eval_percentage, train_file_name, test_file_name, output_dir, class_mapping):
    # create clas dictionary
    class_dict: typing.Dict[str, int] = {}
    for mapping in class_mapping:
        class_dict[mapping[0]] = mapping[1]

    images_path = Path(images_dir)
    output_path = Path(output_dir)
    train_writer = tf.python_io.TFRecordWriter(
        str(output_path / train_file_name))
    test_writer = tf.python_io.TFRecordWriter(
        str(output_path / test_file_name))

    total_train = 0
    total_test = 0

    all_tf_examples = [] # first add all tfexamples into array
    for annotation_path in images_path.glob("**/*.xml"):
        if annotation_path.name == "default.xml":
            continue

        # process one .xml file = one image = one tf.Example
        dfet = xml.etree.ElementTree.parse(annotation_path).getroot()

        fullpath = Path(dfet.find("path").text)
        name = dfet.find("filename").text
        if not fullpath.exists():
            realpath = annotation_path.parent / name
            if realpath.exists():
                print("[W] Invalid file path ({}). Fixing to {}".format(fullpath, realpath))
                fullpath = realpath
            else:
                print("[W] File doesn't exist ({}). Skipping...".format(fullpath))
                continue

        with tf.gfile.GFile(str(fullpath), 'rb') as fid:
            raw_encoded_image=fid.read()
        raw_encoded_image_io=io.BytesIO(raw_encoded_image)
        image = PIL.Image.open(raw_encoded_image_io)
        width, height=image.size

        filename = name.encode('utf8')
        random_name = (''.join(random.choices(
            string.ascii_lowercase + string.digits, k=64)) + ''.join(fullpath.suffixes)).encode('utf8')
        image_format = image.format.lower()
        if image_format != "jpeg" and image_format != "png":
            print("[W] Invalid image format for {}. Should be jpeg/png, is {}. Skipping...".format(fullpath, image_format))
            continue
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        for member in dfet.findall('object'):
            bndbox = member.find("bndbox")
            class_name = member.find("name").text

            xmins.append(int(bndbox.find("xmin").text) / width)
            xmaxs.append(int(bndbox.find("xmax").text) / width)
            ymins.append(int(bndbox.find("ymin").text) / height)
            ymaxs.append(int(bndbox.find("ymax").text) / height)
            classes_text.append(class_name.encode('utf8'))
            classes.append(class_dict[class_name])

        tf_example=tf.train.Example(features = tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(random_name),
            'image/encoded': dataset_util.bytes_feature(raw_encoded_image),
            'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        all_tf_examples.append(tf_example)

    random.shuffle(all_tf_examples) # shuffle for maximum randomness to not get same images in a single batch
    for tf_example in all_tf_examples:
        n = random.randint(1, 100)
        if n <= eval_percentage:
            total_test += 1
            test_writer.write(tf_example.SerializeToString())
        else:
            total_train += 1
            train_writer.write(tf_example.SerializeToString())

    train_writer.close()
    test_writer.close()
    print("Total train images: {}".format(total_train))
    print("Total test images: {}".format(total_test))


@utils_group.command(name = "unify-dataset", help = "Divides the dataaset to train and test, moves training images to one directory and testing images to another and generates corresponding CSV files. These can be used with generate_tfrecord from object_detection")
@click.option("--images-dir", "-i", "images_dir", required = True, help = "Path to saved screenshots.")
@click.option("--train-dir", "train_dir", required = True, help = "Tensorflow train dir")
@click.option("--test-dir", "test_dir", required = True, help = "Tensorflow test dir")
@click.option("--random-n", "-n", "n", type = int, required = False, help = "Random sample of data to use. Defaults to all if not specified.")
def unify_dataset(images_dir, train_dir, test_dir, n):
    train_dir=Path(train_dir)
    test_dir=Path(test_dir)
    all_items=[]

    images_path=Path(images_dir)
    for annotation_path in images_path.glob("**/*.xml"):
        if annotation_path.name == "default.xml":
            continue

        dfet=xml.etree.ElementTree.parse(annotation_path).getroot()

        fullpath=Path(dfet.find("path").text)
        name=dfet.find("filename").text
        random_name=''.join(random.choices(
            string.ascii_lowercase + string.digits, k=32)) + ''.join(fullpath.suffixes)
        size=dfet.find("size")
        width=size.find("width").text
        height=size.find("height").text

        item_strs=[]
        for member in dfet.findall('object'):
            bndbox=member.find("bndbox")
            xmin=bndbox.find("xmin").text
            xmax=bndbox.find("xmax").text
            ymin=bndbox.find("ymin").text
            ymax=bndbox.find("ymax").text

            item_strs.append(
                f"{random_name},{width},{height},screen,{xmin},{ymin},{xmax},{ymax}")
        all_items.append({
            'random_name': random_name,
            'fullpath': fullpath,
            'strs': item_strs,
        })

    random.seed(time.time())
    if n != None and n > 0:
        samples=random.sample(all_items, n)
    else:
        samples=all_items

    train_labels=open(train_dir / "../train_labels.csv", "a")
    test_labels=open(test_dir / "../test_labels.csv", "a")
    header="filename,width,height,class,xmin,ymin,xmax,ymax"
    train_labels.write(header + "\n")
    test_labels.write(header + "\n")
    for item in samples:
        d=''
        f=None
        n=random.randint(1, 100)
        if n <= 30:
            d=test_dir
            f=test_labels
        else:
            d=train_dir
            f=train_labels

        shutil.copy2(item['fullpath'], d / item['random_name'])
        for s in item['strs']:
            f.write(s + "\n")

    train_labels.flush()
    train_labels.close()
    test_labels.flush()
    test_labels.close()


@utils_group.command(name = "annotate", help = "Generate annotations from default.xml for all images in the same directory.")
@click.option("--images-dir", "-i", "images_dir", required = True, help = "Path to saved screenshots.")
@click.option("--force-overwrite", "-f", "force_overwrite", is_flag = True, help = "Overwrite existing annotations for images")
def generate_annotations(images_dir, force_overwrite):
    images_path=Path(images_dir)
    for default_annotation_path in images_path.glob("**/default.xml"):
        session_dir=default_annotation_path.parent
        annotation=xml.etree.ElementTree.parse(
            default_annotation_path).getroot()

        objects = ""
        object_str = """<object>
            <name>screen</name>
            <pose>Unspecified</pose>
            <truncated>1</truncated>
            <difficult>{difficult}</difficult>
            <bndbox>
                <xmin>{xmin}</xmin>
                <ymin>{ymin}</ymin>
                <xmax>{xmax}</xmax>
                <ymax>{ymax}</ymax>
            </bndbox>
        </object>"""

        # find dimensions from either generic PascalVOC format or my custom dump format
        try:
            object_nodes = annotation.findall("object")
            if object_nodes != None:
                for obj in object_nodes:
                    bndbox = obj.find("bndbox")

                    xmin = bndbox.find("xmin").text
                    xmax = bndbox.find("xmax").text
                    ymin = bndbox.find("ymin").text
                    ymax = bndbox.find("ymax").text

                    objects += object_str.format(xmin=xmin,
                                                 ymin=ymin, xmax=xmax, ymax=ymax)
            else:
                xmin = annotation.find("xmin").text
                xmax = annotation.find("xmax").text
                ymin = annotation.find("ymin").text
                ymax = annotation.find("ymax").text
                objects = object_str.format(xmin=xmin,
                                            ymin=ymin, xmax=xmax, ymax=ymax)
        except Exception as e:
            print(
                f"Cannot get dimensions for {default_annotation_path}. Check format. Exception: {e}", file=sys.stderr)
            continue

        # generate annotation for each image
        for screenshot_file in session_dir.glob("*.png"):
            annotation_path = session_dir / \
                (str(screenshot_file.stem) + ".xml")
            if annotation_path.exists() and not force_overwrite:
                continue

            frame = PIL.Image.open(screenshot_file)
            width, height = frame.size
            depth = 3
            frame.close()

            annotation_xml = """<annotation>
        <folder>{folder}</folder>
        <filename>{filename}</filename>
        <path>{fullpath}</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>{width}</width>
            <height>{height}</height>
            <depth>{depth}</depth>
        </size>
        <segmented>0</segmented>
        {objects}
    </annotation>""".format(folder=session_dir.stem, filename=screenshot_file.name, fullpath=screenshot_file, height=height, width=width, depth=depth, objects=objects)
            f = open(annotation_path, "w")
            f.write(annotation_xml)
            f.close()


@utils_group.command("image-to-text", help="OCR images")
@click.option("--bag", "-b", "bag", is_flag=True, type=bool, default=False, help="Show bag of words instead of raw text")
@click.option("--language", "-l", "language", required=False, help="Language of the text in image")
@click.option("--output-file", "-o", "output_file", show_default=True, default="-", type=click.File('w'), help="File to output the csv output. Defaults to stdout")
@click.argument("images", nargs=-1, type=click.Path(exists=True))
def image_to_text(language, images, bag, output_file):
    for image_path in images:
        image = Image(Path(image_path))

        if bag:
            print(image.get_bag_of_words(language), file=output_file)
        else:
            print(image.get_text(language), file=output_file)
        image.show()
        while cv2.waitKey(0) != ord('q'):
            pass
        cv2.destroyAllWindows()

@utils_group.command(name="detect-image")
@click.option("--model-dir", "-m", "model_dir", type=click.Path(exists=True), required=True, help="Path to the directory containing the model's checkpoint")
@click.option("--labelmap-file", "-l", "labelmap_file", type=click.Path(exists=True), required=True, help="Path to the pbtxt labelmap file")
@click.option("--plot", "-p", "plot", type=bool, is_flag=True, default=False, help="Show plotted image with detected objects")
@click.option("--plot-keypoints", "-pk", "plot_keypoints", type=bool, is_flag=True, default=False, help="Detect and plot keypoints in the screen.")
@click.option("--output-file", "-o", "output_file", show_default=True, default="-", type=click.File('w'), help="File to output the csv output. Defaults to stdout")
@click.argument("images", nargs=-1, type=click.Path(exists=True))
def image_detection(model_dir, labelmap_file, images, plot, plot_keypoints, output_file):
    detector = Detector(Path(model_dir), Path(labelmap_file))

    print("file,left,right,top,bottom,score", file=output_file)
    for image_path in images:
        image = Image(image_path)

        (boxes, scores, classes, num) = detector.detect_objects(image)

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        for i in range(boxes.shape[0]):
            if scores[i] < 0.9:
                continue

            box = tuple(boxes[i].tolist())

            ymin, xmin, ymax, xmax = box
            h, w, d = image.image_data.shape

            (left, right, top, bottom) = (xmin * w, xmax * w, ymin * h, ymax * h)
            print("{image_path},{left},{right},{top},{bottom},{score}".format(
                image_path=image_path,
                score=scores[i],
                left=left,
                right=right,
                top=top,
                bottom=bottom), file=output_file
            )

        if plot:
            if plot_keypoints:
                image.get_keypoints_and_descriptors()
                cv2.drawKeypoints(image.image_data, image.keypoints, image.image_data)

            detected = detector.plot_result(image, boxes, scores, classes, num)

            cv2.imshow("Detected", detected)
            while cv2.waitKey(0) != ord('q'):
                pass
            cv2.destroyAllWindows()


@utils_group.command(name="detect-video")
@click.option("--model-dir", "-m", "model_dir", type=click.Path(exists=True), required=True, help="Path to the directory containing the model's checkpoint")
@click.option("--labelmap-file", "-l", "labelmap_file", type=click.Path(exists=True), required=True, help="Path to the pbtxt labelmap file")
@click.option("--skip-first", "-s", "skip_seconds", type=int, default=0, help="How many seconds to skip from the beginning")
@click.option("--every", "-e", "every_seconds", type=int, default=1, help="How many seconds to skip between processed frames", show_default=True)
@click.option("--plot", "-p", "plot", type=bool, is_flag=True, default=False, help="Show plotted image with detected objects")
@click.option("--plot-keypoints", "-pk", "plot_keypoints", type=bool, is_flag=True, default=False, help="Detect and plot keypoints in the screen.")
@click.option("--output-file", "-o", "output_file", show_default=True, default="-", type=click.File('w'), help="File to output the csv output. Defaults to stdout")
@click.argument("video_path", type=click.Path(exists=True))
def video_detection(model_dir, labelmap_file, skip_seconds, every_seconds, plot, plot_keypoints, output_file, video_path):
    video = Video(Path(video_path))
    detector = Detector(Path(model_dir), Path(labelmap_file))

    video.forward(skip_seconds)
    print("file,frame_no,left,right,top,bottom,label,score", file=output_file)
    while True:
        frame_image = video.get_current_frame()
        if frame_image is None:
            break

        (boxes, scores, classes, num) = detector.detect_objects(frame_image)

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        for i in range(boxes.shape[0]):
            if scores[i] < 0.9:
                continue

            box = tuple(boxes[i].tolist())

            ymin, xmin, ymax, xmax = box
            h, w, d = frame_image.image_data.shape

            (left, right, top, bottom) = (xmin * w, xmax * w, ymin * h, ymax * h)
            print("{video_path},{frame_no},{left},{right},{top},{bottom},{score}".format(
                video_path=video_path,
                frame_no=video.get_current_frame_no(),
                score=scores[i],
                left=left,
                right=right,
                top=top,
                bottom=bottom), file=output_file
            )

        if plot:
            detected_frame = detector.plot_result(
                frame_image, boxes, scores, classes, num)
            cv2.imshow("Detected", detected_frame)
            if cv2.waitKey(1) == ord('q'):
                break
        video.forward(every_seconds)

    cv2.destroyAllWindows()

@utils_group.command("convert-pdf", help="Convert PDF to separate images")
@click.option("--source-pdf", "-i", "input_pdf", type=click.Path(exists=True, dir_okay=False), help="Path to the input PDF file")
@click.option("--output-dir", "-o", "output_dir", type=click.Path(exists=True, file_okay=True), help="Path to output the images")
@click.option("--prefix", "-p", "filename_prefix", default="slide", type=str, help="Filename prefix. Images will be name {prefix}-{id}.{extension}")
@click.option("--format", "-f", "file_format", type=str, default="jpg", help="Image format to output", show_default=True)
@click.option("--width", "-w", "output_width", type=int, default=-1, help="Width at which to output the images. Aspect will be kept", show_default=True)
def convert_pdf(input_pdf, output_dir, file_format, filename_prefix, output_width):
    pdf_file = Presentation(input_pdf)
    pdf_file.export(Path(output_dir),
                    file_prefix=filename_prefix, extension=file_format, new_width=output_width)

@utils_group.command("matcher", help="Match best image from a set of candidate images")
@click.option("--keypoints-matching/--no-keypoints-matching", "keypoints_matching", default=True, help="Match slides using keypoints extraction and matching")
@click.option("--text-matching/--no-text-matching", "text_matching", default=True, help="Match slides using text extraction and comparison")
@click.option("--text-weight", "text_weight", type=int, default=2, help="Weight of text matching score in the final match score. Won't be used if not using textual matching", show_default=True)
@click.option("--keypoints-weight", "keypoints_weight", type=int, default=3, help="Weight of keypoints matching score in the final match score. Won't be used if not using keypoints matching", show_default=True)
@click.option("--text-lang", "-l", "language", required=False, help="Language of the text in image")
@click.option("--interactive", "-i", "interactive", type=bool, default=False, is_flag=True, help="Run interactively - use [a] to accept, [r] to reject, [q] to quit")
@click.argument("query_image_path", type=click.Path(exists=True))
@click.argument("candidate_images_paths", nargs=-1, type=click.Path(exists=True))
def matcher(query_image_path, candidate_images_paths, keypoints_matching, text_matching, text_weight, keypoints_weight, language, interactive):
    matcher = Matcher.from_paths(language, *candidate_images_paths)
    query_image = Image(Path(query_image_path))

    match = matcher.match(query_image,
                            do_keypoints=keypoints_matching,
                            do_text=text_matching,
                            keypoints_weight=keypoints_weight,
                            text_weight=text_weight)

    best_match = match.best_match()
    if best_match is None:
        print("NO MATCH", file=sys.stderr)
        return

    print(best_match[0].image_path)
    if interactive:
        # print()
        # print("QUERY WORDS: ", query_image.bag)
        # print("BEST WORDS: ", best_match[0].bag)
        print()
        print("All candidates:")
        for candidate_match in match.candidates:
            print("\t{}\t=> {}".format(
                candidate_match[0].image_path, candidate_match[1]))

        match.plot_best()
        # cv2.imshow("Best match", np.hstack(match_height([
        #     query_image.image_data, best_match[0].image_data], False)))
        # while cv2.waitKey(10) != ord("q"):
        #     pass
        # cv2.destroyAllWindows()
