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

import RPi.GPIO as GPIO

BUZZER_PIN = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)

pwm = GPIO.PWM(BUZZER_PIN, 1)

def play_tone(frequency, duration):
    pwm.ChangeFrequency(frequency)
    pwm.start(50)
    time.sleep(duration)
    pwm.stop()

player_num = 3
print(f"player_num: {player_num}")

boolplayer = [0 for _ in range(player_num)]

pi_camera = VideoCamera()
app = Flask(__name__)
cascade = cv2.CascadeClassifier("/home/aj/fd/haarcascade_frontalface_alt2.xml")

# プレイヤー画像を読み込む
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

    allow_duration = random.uniform(5, 10)
    monitor_duration = random.uniform(3, 8)

    monitoring = False
    last_switch_time = time.time()
    last_buzzer_time = 0

    while True:
        current_time = time.time()
        elapsed = current_time - last_switch_time

        if monitoring and elapsed > monitor_duration:
            monitoring = False
            last_switch_time = current_time
            allow_duration = random.uniform(5, 10)

        elif not monitoring and elapsed > allow_duration:
            monitoring = True
            last_switch_time = current_time
            monitor_duration = random.uniform(3, 8)

        # 許可フェーズ中にブザーを鳴らす
        if not monitoring:
            if current_time - last_buzzer_time > 1.0:
                play_tone(262, 0.3)  # ドの音を0.3秒鳴らす
                last_buzzer_time = current_time

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
                best_player_idx = None

                # 顔検出が成功した場合
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        face_img = gray[y:y + h, x:x + w]

                        if player_faces[0] is not None:
                            # 顔画像をテンプレートサイズに合わせる
                            face_img_resized = cv2.resize(
                                face_img,
                                (player_faces[0].shape[1], player_faces[0].shape[0])
                            )

                            for idx, ref_img in enumerate(player_faces):
                                if ref_img is None:
                                    continue

                                h_f, w_f = face_img_resized.shape
                                h_r, w_r = ref_img.shape

                                # face_img_resized が大きすぎる場合はリサイズする
                                if h_f > h_r or w_f > w_r:
                                    face_img_resized = cv2.resize(face_img_resized, (w_r, h_r))

                                # matchTemplate の入力順序に注意
                                res = cv2.matchTemplate(ref_img, face_img_resized, cv2.TM_CCOEFF_NORMED)
                                score = res[0][0]

                                print(f"Player {idx} 類似度: {score:.3f}")

                                if score > best_match_score:
                                    best_match_score = score
                                    best_player_idx = idx

                else:
                    # 顔検出失敗 → 全体画像で比較
                    face_img = gray

                    if player_faces[0] is not None:
                        face_img_resized = cv2.resize(
                            face_img,
                            (player_faces[0].shape[1], player_faces[0].shape[0])
                        )

                        for idx, ref_img in enumerate(player_faces):
                            if ref_img is None:
                                continue

                            h_f, w_f = face_img_resized.shape
                            h_r, w_r = ref_img.shape

                            if h_f > h_r or w_f > w_r:
                                face_img_resized = cv2.resize(face_img_resized, (w_r, h_r))

                            res = cv2.matchTemplate(ref_img, face_img_resized, cv2.TM_CCOEFF_NORMED)
                            score = res[0][0]

                            print(f"Player {idx} 類似度(全体比較): {score:.3f}")

                            if score > best_match_score:
                                best_match_score = score
                                best_player_idx = idx

                # 類似度が一定以上ならプレイヤー確定
                if best_match_score > 0.3 and best_player_idx is not None:
                    identified_player = best_player_idx

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
                    print("誰も特定できませんでした。")

            ret, jpeg = cv2.imencode(".jpg", orig_frame)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(pi_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        GPIO.cleanup()