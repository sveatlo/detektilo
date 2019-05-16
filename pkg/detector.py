# autopep8: skip-file

import sys
import os
import typing
from pathlib import Path

import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt

from .image import Image


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class Detector():
    def __init__(self, model_dir: Path, labelmap_file: Path):
        print("[I] Initializing Detector...")
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(str(model_dir / 'frozen_inference_graph.pb'), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(
                graph=self.detection_graph,
                config=tf.ConfigProto(
                    allow_soft_placement=True,
                    # log_device_placement=True
                )
            )
        self.label_map = label_map_util.load_labelmap(str(labelmap_file))
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map, max_num_classes=100, use_display_name=True)
        self.category_index = label_map_util.create_category_index(
            self.categories)
        self.image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name(
            'detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')

    def detect_objects(self, image: Image) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        image_expanded = np.expand_dims(image.image_data, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})
        return (boxes, scores, classes, num)

    def plot_result(self, image: Image, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, num: np.ndarray) -> np.ndarray:
        image_data_copy = image.image_data.copy()
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_data_copy,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.9)

        return image_data_copy
