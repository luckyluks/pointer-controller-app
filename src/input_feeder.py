'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import os
import cv2
import sys

import logging as log
from numpy import ndarray

class InputFeeder:
    def __init__(self, input_file=None):
        '''
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        # Determine if file exists
        if (not os.path.exists(os.path.abspath(input_file))) and ("cam" not in input_file):
            raise FileNotFoundError(f"Input file \"{input_file}\" does not exist.")
        
        self.input_file=input_file

        # Determine file type
        input_argument = self.input_file.lower()
        if input_argument.endswith((".jpg", ".bmp", ".png")):
            self.input_type = "image"
        elif input_argument.endswith((".mp4", ".avi")):
            self.input_type = "video"
        elif "cam" in input_argument:
            self.input_type = "cam"
        else:
            log.warn("File type not supported: {}".format(self.input_file))
            sys.exit("ERROR: unknown input file type!")
    
    def load_data(self):
        # Open correct input stream
        if self.input_type == "video":
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type == "cam":
            self.cap=cv2.VideoCapture(0)
        elif self.input_type == "image":
            self.cap=cv2.imread(self.input_file)
        else:
            log.error(f"Given Source \"{self.input_file}\" (type: {self.input_type}) not supported!")
            sys.exit()
        
        # Gather meta data 
        initial_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        initial_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        log.info(f"Loaded input stream: \"{self.input_file}\" (type:{self.input_type}, fps:{fps}, format:{(initial_w, initial_h)})")

        return fps, (initial_w, initial_h)

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        while True:
            for _ in range(1):
                _, frame=self.cap.read()

            yield frame


    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()
            cv2.destroyAllWindows()

