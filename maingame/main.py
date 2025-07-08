#https://www.mikan-tech.net/entry/raspi-speaker 音声を流す参考URL
#md.pyが元になっている
from flask import Flask, render_template, Response, request, send_from_directory
import os
import cv2
from camera import VideoCamera
import imutils
#from pydub import AudioSegment
#from pydub.playback import play
import time
import threading
import random

#音性を流す
"""
sound = AudioSegment.from_mp3("hoge.mp3")
play(sound)
"""
#player_num = int(input("プレイヤーの人数を入力してください"))
player_num = 3
print(player_num)

boolplayer = []

for i in range(player_num):
    boolplayer.append(0)

# Globals (do not edit)
pi_camera = VideoCamera() 
app = Flask(__name__)
cascade = cv2.CascadeClassifier("/home/aj/fd/haarcascade_frontalface_alt2.xml")

motion_detected_frames = 0
motion_detected_threshold = 3

shared_state = {"num": 1}
lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html') #you can customze index.html here
@app.route('/playersit')
def playersit():
    playerbool=""
    for i in boolplayer:
        playerbool += str(i)
        playerbool += "\n"
    return playerbool

def gen(camera,num):
    frame_count = 0
    consecutive_frame = 4
    while True:
        with lock:
            num = shared_state["num"]

        if num==0:
            print('基準画像設定')
            background = camera.get_frame()
            # convert the background model to grayscale format
            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        #num=(num+1)%100
      
        frame = camera.get_frame()
        frame_count += 1
        orig_frame = frame.copy()

        # IMPORTANT STEP: convert the frame to grayscale first
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % consecutive_frame == 0 or frame_count == 1:
            frame_diff_list = []
        
        # find the difference between current frame and base frame
        frame_diff = cv2.absdiff(gray, background)
        # thresholding to convert the frame to binary
        ret, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        # dilate the frame a bit to get some more white area...
        # ... makes the detection of contours a bit easier
        dilate_frame = cv2.dilate(thres, None, iterations=2)
        # append the final result into the `frame_diff_list`
        frame_diff_list.append(dilate_frame)
        
        # if we have reached `consecutive_frame` number of frames
        if len(frame_diff_list) == consecutive_frame:
            # add all the frames in the `frame_diff_list`
            sum_frames = sum(frame_diff_list)
        
            # find the contours around the white segmented areas
            contours, hierarchy = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # draw the contours, not strictly necessary
            for i, cnt in enumerate(contours):
                cv2.drawContours(frame, contours, i, (0, 0, 255), 3)

            for contour in contours:
                # continue through the loop if contour area is less than 1500...
                # ... helps in removing noise detection

                if cv2.contourArea(contour) >  1500:
                    motion_detected_frames += 1
                else:
                    motion_detected_frames = 0

                if motion_detected_frames >= motion_detected_threshold:
                  # get the xmin, ymin, width, and height coordinates from the contours
                    (x, y, w, h) = cv2.boundingRect(contour)
                    # 動体検出領域を切り出す
                    roi = orig_frame[y:y+h, x:x+w]

                    # グレースケールに変換
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                    # 顔検出
                    faces = cascade.detectMultiScale(roi_gray,scaleFactor=1.1,minNeighbors=3,minSize=(30, 30))

                    if len(faces) > 0:
                        print("人間を検知！")

                    # ROI内の顔矩形を元の座標系に変換して描画
                        for (fx, fy, fw, fh) in faces:
                            # ROI内座標 → 全体座標へ
                            cv2.rectangle(orig_frame,
                                (x + fx, y + fy),
                                (x + fx + fw, y + fy + fh),
                                (0, 0, 255),
                                2
                            )
                    else:
                        print("この人でなし!")
                  # draw the bounding boxes
                  #cv2.rectangle(frame_diff, (x, y), (x+w, y+h), (255, 255, 255), 2)
                    cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
            #cv2.imshow('Detected Objects', frame_diff)
            #cv2.imshow('Detected Objects', sum_frames)
            #cv2.imshow('orig_frame', orig_frame)

            ret, jpeg = cv2.imencode(".jpg", orig_frame)
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
def control_loop():
    while True:
        sleep_time = random.randint(5, 10)
        print(f"停止中 {sleep_time}秒")
        time.sleep(sleep_time)

        print("基準画像設定モードへ")
        with lock:
            shared_state["num"] = 0

        # num=0 の処理が終わるまで少し待つ
        time.sleep(2) # 必要に応じて調整

        active_time = 5
        print(f"num=1 で稼働開始 {active_time}秒")
        with lock:
            shared_state["num"] = 1

        time.sleep(active_time)
        
@app.route('/video_feed')
def video_feed():
    return Response(gen(pi_camera,0),mimetype='multipart/x-mixed-replace; boundary=frame')
"""
if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=False)
"""

if __name__ == "__main__":
    t = threading.Thread(target=control_loop)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=True)