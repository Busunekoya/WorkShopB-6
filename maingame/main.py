#https://www.mikan-tech.net/entry/raspi-speaker 音声を流す参考URL
#md.pyが元になっている
from flask import Flask, render_template, Response, request, send_from_directory
import os
import cv2
from camera import VideoCamera
import imutils
# from pydub import AudioSegment
# from pydub.playback import play
import threading
import random
import time
import requests

player_num = 3
print(player_num)

boolplayer = []
for i in range(player_num):
    boolplayer.append(0)

# Globals
pi_camera = VideoCamera()
app = Flask(__name__)
cascade = cv2.CascadeClassifier("/home/aj/fd/haarcascade_frontalface_alt2.xml")

# motion 検出用
motion_detected_frames = 0
motion_detected_threshold = 3
motion_pixel_threshold = 5000   # 動きありと判定する総画素数の閾値

shared_state = {"num": 1}
lock = threading.Lock()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/playersit')
def playersit():
    playerbool = ""
    for i in boolplayer:
        playerbool += str(i)
        playerbool += "\n"
    return playerbool


def gen(camera):
    frame_count = 0
    consecutive_frame = 4
    num = 0

    # 監視・許可サイクルの設定（秒単位）
    allow_duration = 5      # 動いても無視する時間
    monitor_duration = 3    # 動いたら検知する時間

    # 状態管理
    monitoring = False
    last_switch_time = time.time()

    while True:
        # サイクル管理
        current_time = time.time()
        elapsed = current_time - last_switch_time

        if monitoring and elapsed > monitor_duration:
            # 監視時間終了 → 許可時間へ
            monitoring = False
            last_switch_time = current_time
            print("★ 許可時間に切り替え（動いてもOK）")

        elif not monitoring and elapsed > allow_duration:
            # 許可時間終了 → 監視時間へ
            monitoring = True
            last_switch_time = current_time
            print("★ 監視時間に切り替え（動いたらダメ！）")

        if num == 0:
            print('基準画像設定')
            background = camera.get_frame()
            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        num = (num + 1) % 100

        frame = camera.get_frame()
        frame_count += 1
        orig_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % consecutive_frame == 0 or frame_count == 1:
            frame_diff_list = []

        frame_diff = cv2.absdiff(gray, background)
        ret, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        dilate_frame = cv2.dilate(thres, None, iterations=2)
        frame_diff_list.append(dilate_frame)

        if len(frame_diff_list) == consecutive_frame:
            sum_frames = sum(frame_diff_list)

            contours, hierarchy = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False

            for contour in contours:
                if cv2.contourArea(contour) > 1500:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    motion_detected = True

            # ★ 監視時間中に動きを検知したら信号を送る
            if monitoring and motion_detected:
                print("！！！動いた！/playersit を呼びます！！！")

                # 例：ランダムに誰かを座らせる
                random_index = random.randint(0, player_num - 1)
                boolplayer[random_index] = 1

                try:
                    res = requests.get("http://127.0.0.1:5000/playersit")
                    print("playersit のレスポンス:")
                    print(res.text)
                except Exception as e:
                    print("playersit呼び出し失敗:", e)

            ret, jpeg = cv2.imencode(".jpg", orig_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(pi_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)