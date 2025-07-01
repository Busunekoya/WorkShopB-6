"""import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setup(25, GPIO.OUT)

gameset = 0



def darmagame():
    if gameset == 0:
        GPIO.output(25, GPIO.HIGH)
        print('light ON')
        GPIO.output(25, GPIO.LOW)
"""
player_num = int(input("プレイヤーの人数を入力してください"))
print(player_num)

boolplayer = []

for i in range(player_num):
    boolplayer.append(0)

print(boolplayer)