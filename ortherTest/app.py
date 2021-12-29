from flask import Flask, render_template, Response

# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run pose classification and pose estimation."""
import argparse
import logging
import sys
import time
import multiprocessing as mp
import cv2
from ml import Classifier
from ml import Movenet
from ml import MoveNetMultiPose
from ml import Posenet
import utils


def run(estimation_model: str, tracker_type: str, classification_model: str,
        label_file: str, camera_id: int, width: int, height: int) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
      estimation_model: Name of the TFLite pose estimation model.
      tracker_type: Type of Tracker('keypoint' or 'bounding_box').
      classification_model: Name of the TFLite pose classification model.
        (Optional)
      label_file: Path to the label file for the pose classification model. Class
        names are listed one name per line, in the same order as in the
        classification model output. See an example in the yoga_labels.txt file.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
    """
    print("run方法进行中.....")
    # Notify users that tracker is only enabled for MoveNet MultiPose model.
    if tracker_type and (estimation_model != 'movenet_multipose'):
        logging.warning(
            'No tracker will be used as tracker can only be enabled for '
            'MoveNet MultiPose model.')

    # Initialize the pose estimator selected.
    pose_detector = Movenet(estimation_model)

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    max_detection_results = 3
    fps_avg_frame_count = 10

    # ================================多核使用====================================
    q = mp.Queue()
    # initial frames and processes
    # 想要几个进程处理图片就用几组，但并不是用的越多越好，
    # 树莓派的CPU一共有4个核，全部用上可能会影响其他的性能，自己试的时候2个会好一点。
    ret, image1 = cap.read()
    image1 = cv2.resize(image1, (int(width * 0.3), int(height * 0.3)))
    res1 = mp.Process(target=modelTrain, args=(pose_detector, image1,q))
    res1.start()

    # 可以把识别用的图片像素缩小，可以加快速度，同时也可以减少cpu负担，
    # 然后再把识别框扩大相应倍数，在原图片上放出。
    ret, image2 = cap.read()
    image2 = cv2.resize(image2, (int(width * 0.3), int(height * 0.3)))
    res2 = mp.Process(target=modelTrain, args=(pose_detector, image2,q))
    res2.start()

    # ================================多核使用====================================

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        print("检验视频流中....")
        counter += 1

        # process 1
        # 如果想要逐帧处理，那就用pass，如果想跳帧就选择上面两句
        if (res1.is_alive()):
            # 从队列中获取图片 修正 输出
            image = q.get()
            image = cv2.resize(image, (width, height))
            # Calculate the FPS
            if counter % fps_avg_frame_count == 0:
                end_time = time.time()
                fps = fps_avg_frame_count / (end_time - start_time)
                start_time = time.time()

            # Show the FPS
            fps_text = 'FPS = ' + str(int(fps))
            text_location = (left_margin, row_size)
            cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        font_size, text_color, font_thickness)

            ret, jpeg = cv2.imencode('.jpg', image)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            ret, image1 = cap.read()
        else:
            ret, image1 = cap.read()
            image1 = cv2.resize(image1, (int(width * 0.3), int(height * 0.3)))
        res1 = mp.Process(target=modelTrain, args=(pose_detector, image1, q))
        res1.start()

        # process 2
        # 如果想要逐帧处理，那就用pass，如果想跳帧就选择上面两句
        if (res2.is_alive()):
            # 从队列中获取图片 修正 输出
            image = q.get()
            image = cv2.resize(image, (width, height))
            # Calculate the FPS
            if counter % fps_avg_frame_count == 0:
                end_time = time.time()
                fps = fps_avg_frame_count / (end_time - start_time)
                start_time = time.time()

            # Show the FPS
            fps_text = 'FPS = ' + str(int(fps))
            text_location = (left_margin, row_size)
            cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        font_size, text_color, font_thickness)

            ret, jpeg = cv2.imencode('.jpg', image)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            ret, image2 = cap.read()
        else:
            ret, image2 = cap.read()
            image2 = cv2.resize(image2, (int(width * 0.3), int(height * 0.3)))
        res2 = mp.Process(target=modelTrain, args=(pose_detector, image2, q))
        res2.start()



def modelTrain(pose_detector, image,q):
    # Run pose estimation using a SinglePose model, and wrap the result in an
    # array.
    list_persons = [pose_detector.detect(image)]

    # Draw keypoints and edges on input image
    image = utils.visualize(image, list_persons)
    q.put(image)


app = Flask(__name__, static_folder='./static')


@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')


@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of estimation model.',
        required=False,
        default='movenet_lightning')
    parser.add_argument(
        '--tracker',
        help='Type of tracker to track poses across frames.',
        required=False,
        default='bounding_box')
    parser.add_argument(
        '--classifier', help='Name of classification model.', required=False)
    parser.add_argument(
        '--label_file',
        help='Label file for classification.',
        required=False,
        default='labels.txt')
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        default=550)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        default=300)
    args = parser.parse_args()
    print("video_feed方法完成中....")
    return Response(run(args.model, args.tracker, args.classifier, args.label_file,
                        int(args.cameraId), args.frameWidth, args.frameHeight),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
