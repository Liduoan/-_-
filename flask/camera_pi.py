import io
import time
import picamera
import cv2
import numpy as np

import utils
from base_camera import BaseCamera
from ml import Movenet


class Camera(BaseCamera):
    @staticmethod
    def frames():
        with picamera.PiCamera() as camera:
            # let camera warm up
            time.sleep(2)
            print("导入模型...........")
            pose_detector = Movenet("movenet_lightning")

            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, 'jpeg',
                                                 use_video_port=True):

                data = np.fromstring(stream.getvalue(), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.CV_LOAD_IMAGE_UNCHANGED)
                # ==============识别算法================
                print("识别图片...........")
                list_persons = [pose_detector.detect(image)]
                # Draw keypoints and edges on input image
                image = utils.visualize(image, list_persons)
                # ==============识别算法================

                # return current frame
                stream.seek(0)
                # 输出回顾
                yield cv2.imencode('.jpg', image)[1].tobytes()

                # reset stream for next frame
                stream.seek(0)
                stream.truncate()
