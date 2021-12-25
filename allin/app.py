from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
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
def process_frame(img):
    # BGR转RGB
    # img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_mode.process(img)
    mp_drawing.draw_landmarks(
        img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)
    return img

class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        image = process_frame(image)
        # 在这里处理视频帧
        cv2.putText(image, "hello", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


app = Flask(__name__, static_folder='./static')


@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)