from flask import Flask, render_template, Response, request, send_from_directory
import os
import cv2
from camera import VideoCamera
import imutils

# Globals (do not edit)
pi_camera = VideoCamera() 
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') #you can customze index.html here

def gen(camera):
    cascade = cv2.CascadeClassifier("/home/aj/fd/haarcascade_frontalface_alt2.xml")
    while True:
        frame = camera.get_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
        for (x,y,width,height) in faces:
            cv2.rectangle(frame,(x,y),(x+width,y+height),(255,0,0),2)

        ret, jpeg = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        
@app.route('/video_feed')
def video_feed():
    return Response(gen(pi_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=False)
