import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setup(25, GPIO.OUT)

while True:
    GPIO.output(25, GPIO.HIGH)
    print('light ON')
    sleep(0.5)
    print('light OFF')
    GPIO.output(25, GPIO.LOW)
    sleep(0.5)
