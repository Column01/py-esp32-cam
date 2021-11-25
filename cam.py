import collections
import datetime
import enum
import threading
import time
from typing import Tuple

import cv2
import numpy as np
import requests


class FrameSize(enum.Enum):
    """ The resolution for the camera """

    FRAMESIZE_96X96   = 0     #   96x96
    FRAMESIZE_QQVGA   = 1     #  160x120
    FRAMESIZE_QCIF    = 2     #  176x144
    FRAMESIZE_HQVG    = 3     #  240x176
    FRAMESIZE_240X240 = 4     #  240x240
    FRAMESIZE_QVGA    = 5     #  320x240
    FRAMESIZE_CIF     = 6     #  400x296
    FRAMESIZE_HVGA    = 7     #  480x320
    FRAMESIZE_VGA     = 8     #  640x480
    FRAMESIZE_SVGA    = 9     #  800x600
    FRAMESIZE_XGA     = 10    # 1024x768
    FRAMESIZE_HD      = 11    # 1280x720
    FRAMESIZE_SXGA    = 12    # 1280x1024
    FRAMESIZE_UXGA    = 13    # 1600x1200
    # 3MP Sensors
    FRAMESIZE_FHD     = 14    # 1920x1080
    FRAMESIZE_P_HD    = 15    #  720x1280
    FRAMESIZE_P_3MP   = 16    #  864x1536
    FRAMESIZE_QXGA    = 17    # 2048x1536
    # 5MP Sensors
    FRAMESIZE_QHD     = 18    # 2560x1440
    FRAMESIZE_WQXGA   = 19    # 2560x1600
    FRAMESIZE_P_FHD   = 20    # 1080x1920
    FRAMESIZE_QSXGA   = 21    # 2560x1920


class ESP32SecurityCam(threading.Thread):
    def __init__(self, cam_address: str, framesize: FrameSize, frame_flips: Tuple[bool, bool], capture_length: int, after_event_length: int, do_face_detect: bool):
        """A threaded system to connect to a ESP32 camera module and capture frames when detections are found

        Args:
            cam_address (str): The IP address of the camera that looks like: "http:#192.168.0.123"
            framesize (FrameSize): The framesize, see above enum for resolutions
            frame_flips (Tuple(bool, bool)): A tuple of whether vert and horizontal flip should occur "(Verticle Flip, Horizontal Flip)"
            capture_length (int): The capture length when an event is triggerred in number frames
            after_event_length (int): How many frames after an event you want to record
            do_face_detect (bool): Whether to do facial detection or not
        """
        # Store arguments
        self.cam_address = cam_address
        self.framesize = framesize.value
        self.vert_flip, self.horizontal_flip = frame_flips[0], frame_flips[1]

        self.capture_length = capture_length
        self.after_event_length = after_event_length
        self.do_face_detect = do_face_detect
        # Build URLs from camera address
        self.control_url = self.cam_address + "/control"
        cam_url = cam_address + ":81/stream"
        # Open a video stream with the camera
        self.cap = cv2.VideoCapture(cam_url)
        # Load facial detection model
        self.frontal = cv2.CascadeClassifier('models/frontalface.xml')
        # self.profile = cv2.CascadeClassifier('models/profileface.xml')
        # self.body = cv2.CascadeClassifier('models/fullbody.xml')
        # The current camera frame (used for streaming to flask web server)
        self.cur_frame = None
        # Initialize an FPS variable
        self.fps = 0
        self.fps_snapshots = collections.deque(maxlen=capture_length)
        # Initialize a frame buffer to record some video when faces are detected. capture_length x fps = Video length (seconds) 
        self.frames = collections.deque(maxlen=capture_length)

        self.recording = False
        self.fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.writer = None
        # How many frames after we started recording have been captured
        self.extra_frames = 0
        # Cooldown variable so we don't record constantly
        self.frames_till_record = 0

        self.vid_write_thread = None

        # Initialize the class as a thread
        threading.Thread.__init__(self)

    def run(self):
        # Setup the camera's output
        self.setup_camera()
        ret, frame = self.cap.read() if self.cap.isOpened() else False, False

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if ret:
            while True:
                # Gather start time
                start = time.time()
                # Capture a frame
                ret, frame = self.cap.read()
                # Calculate how long it took to gather the frame
                elapsed = time.time() - start
            
                if ret and frame is not None:
                    # Calculate the current "fps" for this frame and save it
                    fps = 1 // elapsed
                    self.fps_snapshots.append(fps)
                    avg_fps = sum(self.fps_snapshots) // len(self.fps_snapshots)
                    print(f"Frame/fps buffer len: {len(self.frames)} {len(self.fps_snapshots)} Cur/Avg FPS: {fps}/{avg_fps}")
                    if self.do_face_detect:
                        # Convert the frame to gray
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        # Find any faces in frame
                        detections = self.frontal.detectMultiScale(gray, 1.3, 5)

                        # If we have a detection, are not recording, and not on cooldown to record start a recording
                        if detections and not self.recording and self.frames_till_record == 0:
                            for (x, y, w, h) in detections:
                                # highlight the detected faces in the frame
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            print("STARTING RECORDING")
                            self.recording = True
                            # Calculate the average fps for the current frame buffer
                            # This isn't perfect but it gives much better output results than if we use the reported FPS
                            avg_fps = int(sum(self.fps_snapshots) / len(self.fps_snapshots))
                            
                            if self.writer is None:
                                h, w = frame.shape[:2]
                                print(f"VIDEO STATS: {h}p{avg_fps}")
                                file_name = "recording-" + datetime.datetime.now().strftime("%d-%m-%Y %I-%M-%S%p") + ".avi"
                                self.writer = cv2.VideoWriter(file_name, self.fourcc, avg_fps, (w, h), True)

                        # If there is a frames till record cooldown, decrement it
                        if self.frames_till_record > 0:
                            self.frames_till_record -= 1

                    # Store the current frame
                    self.cur_frame = np.copy(frame)
                    self.frames.append(frame)

                    # If we are recording
                    if self.recording:
                        # Increment the captured frames amount
                        self.extra_frames += 1

                        # If we've recorded enough frames after we triggerred the record
                        if self.extra_frames > self.after_event_length:
                            # And we are not already saving frames
                            if self.vid_write_thread is None:
                                # Start saving a new video in a thread with a copy of the frames
                                print(f"Saving frames: {len(self.frames)}")
                                frames_copy = self.frames.copy()
                                self.vid_write_thread = threading.Thread(target=self.save_frame_buffer, args=(frames_copy, ))
                                self.vid_write_thread.start()
                            # If the writing thread is done
                            elif not self.vid_write_thread.is_alive():
                                # Video has been written to disk, we can reset the variables to the original state
                                self.vid_write_thread = None
                                self.writer = None
                                self.recording = False
                                self.extra_frames = 0
                                self.frames_till_record = self.after_event_length

                    k = cv2.waitKey(1) & 0xFF
                    if k == 27:  # esc key ends process
                        self.cap.release()
                        break
                else:
                    print("Camera feed potentially dropped, trying to re-initialize")
                    # Camera may have restarted, try to re-initialize it.
                    try:
                        self.setup_camera()
                        # Pause for 3 seconds to ensure we don't just spam a non-existing endpoint+
                        time.sleep(3)
                    except BaseException:
                        # Ignore errors for now
                        pass

    def setup_camera(self):
        vert_flip = 1 if self.vert_flip else 0
        horizontal_flip = 1 if self.horizontal_flip else 0
        # Set resolution
        resp = requests.get(self.control_url + f"?var=framesize&val={self.framesize}")
        # Set vertical flip
        resp1 = requests.get(self.control_url + f"?var=vflip&val={vert_flip}")
        # Set horizontal flip
        resp2 = requests.get(self.control_url + f"?var=hflip&val={horizontal_flip}")

    def get_frame(self):
        _, jpeg = cv2.imencode('.jpg', self.cur_frame)
        return jpeg.tobytes()

    def save_frame_buffer(self, frames):
        print("Saving frame buffer to disk")
        # Write all frames stored in our buffer to the file
        for frame in frames:
            self.writer.write(frame)
        
        # Release the writer
        self.writer.release()


if __name__ == "__main__":
    cam = ESP32SecurityCam("http:#192.168.0.27", 11, True, False, 100, 50, False)
    cam.start()
