"""
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
    app.run(host='0.0.0.0', port=5000, debug=False)"""

from flask import Flask, render_template, Response, request, send_from_directory
import os
import cv2
from camera import VideoCamera
import imutils
import threading
import random
import time
import requests
import numpy as np

player_num = 3
print(f"player_num: {player_num}")

# 各プレイヤーの状態（0=座ってない、1=座った）
boolplayer = [0 for _ in range(player_num)]

# Flaskアプリ初期化
pi_camera = VideoCamera()
app = Flask(__name__)
cascade = cv2.CascadeClassifier("/home/aj/fd/haarcascade_frontalface_alt2.xml")

# 顔画像の事前登録
player_faces = []
for i in range(player_num):
    path = f"players/{i}.jpg"
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        player_faces.append(img)
        print(f"Loaded player {i} face image: {path}")
    else:
        player_faces.append(None)
        print(f"Player {i} image not found: {path}")

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

    allow_duration = 5      # 動いても無視する時間
    monitor_duration = 3    # 動いたら検知する時間

    monitoring = False
    last_switch_time = time.time()

    while True:
        # 許可/監視サイクルの管理
        current_time = time.time()
        elapsed = current_time - last_switch_time

        if monitoring and elapsed > monitor_duration:
            monitoring = False
            last_switch_time = current_time
            print("★ 許可時間に切り替え（動いてもOK）")

        elif not monitoring and elapsed > allow_duration:
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

            if monitoring and motion_detected:
                print("！！！監視フェーズで動きを検知！顔認識を行います！！！")

                faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

                identified_player = None
                best_match_score = 0

                for (x, y, w, h) in faces:
                    face_img = gray[y:y+h, x:x+w]

                    # 登録画像と同じサイズにリサイズ
                    if player_faces[0] is not None:
                        face_img_resized = cv2.resize(
                            face_img,
                            (player_faces[0].shape[1], player_faces[0].shape[0])
                        )

                        for idx, ref_img in enumerate(player_faces):
                            if ref_img is None:
                                continue

                            res = cv2.matchTemplate(face_img_resized, ref_img, cv2.TM_CCOEFF_NORMED)
                            score = res[0][0]

                            print(f"Player {idx} 類似度: {score:.3f}")

                            if score > best_match_score and score > 0.5:
                                best_match_score = score
                                identified_player = idx

                if identified_player is not None:
                    boolplayer[identified_player] = 1
                    print(f"★★ Player {identified_player} が監視フェーズ中に動きました！")

                    try:
                        res = requests.get("http://127.0.0.1:5000/playersit")
                        print("playersit のレスポンス:")
                        print(res.text)
                    except Exception as e:
                        print("playersit呼び出し失敗:", e)
                else:
                    print("監視フェーズ中に顔認識できず。スキップ。")

            ret, jpeg = cv2.imencode(".jpg", orig_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(pi_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)