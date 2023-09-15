import numpy as np
from ctypes import windll
import time
import threading
import pygame
import glob


class StageInput:
    def __init__(self):
        self.port = 0x4FF8  # LTP
        self.io = windll.LoadLibrary('./inpoutx64.dll')
        self.stage_dict = {'W/N1': 201, 'N2': 202, 'N3': 203, 'R': 204}

    def writeStage(self, stage):
        def func(stage):
            self.io.DlPortWritePortUchar(self.port, self.stage_dict[stage])
            time.sleep(0.01)
            self.io.DlPortWritePortUchar(self.port, 0)

        f = threading.Thread(target=func, args=(stage,))
        f.start()


class AudioInput:
    def __init__(self):
        self.audioPaths = sorted(glob.glob("Audio/*"))
        self.port = 0x4FF8  # LTP
        self.io = windll.LoadLibrary('./inpoutx64.dll')
        self.audio_dict = {"alarm.wav": 1, "apple.wav": 2, "ball.wav": 3, "book.wav": 4, "box.wav": 5,
                           "chair.wav": 6, "kiwi.wav": 7, "microphone.wav": 8, "motorcycle.wav": 9, "pepper.wav": 10,
                           "sheep.wav": 11, "shoes.wav": 12, "strawberry.wav": 13, "tomato.wav": 14, "watch.wav": 15}
        pygame.mixer.init()

    def audioPlay(self):
        def func():
            N = np.random.randint(0, 15)
            path = self.audioPaths[N]

            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(0.2)
            pygame.mixer.music.play()

            # cue同时输入会导致只有一个cue被写入。为了保证CUE的标记可以打到脑电上，声音标记在脑电上将会延迟10ms
            time.sleep(0.01)
            self.io.DlPortWritePortUchar(self.port, self.audio_dict[path.split("\\")[1]])
            time.sleep(0.01)
            self.io.DlPortWritePortUchar(self.port, 0)
            print("Play Sound %s" % (path.split("\\")[1].split(".")[0]))

        f = threading.Thread(target=func(), args=())
        f.start()
        f.join()
