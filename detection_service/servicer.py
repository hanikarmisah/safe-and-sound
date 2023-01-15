from detection_service.utils import *
import logging
import time
import torch
from ovms_helper import ovms_helper
import numpy as np
from typing import List

# TODO: add logger, add time profiling

class Detect_Service():
    def __init__(self, model_name: str, inference_url: str, input_width: int = 640, input_height: int = 640,
                    score_threshold: float = 0.6, nms_threshold: float = 0.3, blackout_threshold: float = 0.6):
        self.inference_url = inference_url
        self.input_width = input_width
        self.input_height = input_height
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.blackout_threshold = blackout_threshold
        self.ovms_helper_obj = ovms_helper(model_name, inference_url)

    def _preprocess_frame(self, frame: np.array):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame, _, _, _ = letterbox(frame, new_shape=(self.input_height, self.input_width))
        frame = frame.transpose(2,0,1).reshape(1, 3, self.input_height, self.input_width)
        frame = frame.astype(np.float32)
        frame /= 255.0
        return frame

    def detect_object(self, frame: np.array):
        frame_height, frame_width, _ = frame.shape
        frame = self._preprocess_frame(frame)
        outputs = self.ovms_helper_obj._predict(frame)
        processed_output = self._postprocess_frame(outputs, frame_width, frame_height)
        return processed_output

    def _postprocess_frame(self, outputs: List, frame_width: int, frame_height: int):
        outputs = torch.Tensor(outputs)
        outputs = non_max_suppression(outputs, self.score_threshold, self.nms_threshold, multi_label=True)
        detections = outputs[0]
        detections[:, :4] = scale_coords([self.input_height, self.input_width], detections[:, :4], [frame_height, frame_width]).round()
        detections = detections.numpy()
        return detections

    def detect_blackout(self, frame: np.array):
        is_blackout = False
        frame_height, frame_width, num_channel = frame.shape
        total_pixels = frame_height * frame_width * num_channel
        black_pixels = np.sum(frame == 0)
        blackout_percent = black_pixels / total_pixels
        if blackout_percent >= self.blackout_threshold:
            is_blackout = True
        return is_blackout

