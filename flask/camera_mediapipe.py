# -*- coding: UTF-8 -*-
import os
import cv2
import mediapipe as mp

from base_camera import BaseCamera


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

        # 导入solution
        mp_pose = mp.solutions.pose

        # 导入绘图函数
        mp_drawing = mp.solutions.drawing_utils
        # 参数：1、颜色，2、线条粗细，3、点的半径
        DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 3, 1)
        DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 3, 1)

        # 导入模型
        pose_mode = mp_pose.Pose(smooth_landmarks=True,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
        while True:
            # read current frame
            _, img = camera.read()
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose_mode.process(img)
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
