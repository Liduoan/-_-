# -*- coding: UTF-8 -*-
import os
import cv2

import utils
from base_camera import BaseCamera
from ml import Movenet


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        width = 380
        height = 300
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # print("导入模型...........")
        pose_detector = Movenet("movenet_lightning")
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')
        # flag = 1
        while True:
            # read current frame
            _, img = camera.read()
            img = cv2.resize(img, (int(width * 0.8), int(height * 0.8)))
            # if flag == 1:
            #     flag = 0
            # else:
            # ==============识别算法================
            list_persons = [pose_detector.detect(img)]
            # print("识别图片...........")
            # print(list_persons[0].score)
            # Draw keypoints and edges on input image
            if list_persons[0].score > 0.35:
                img = utils.visualize(img, list_persons)
            # ==============识别算法================
            # flag = 1
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
