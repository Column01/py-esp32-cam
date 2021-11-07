from threading import Thread

import cv2


class ESP32SecurityCam(Thread):
    def __init__(self, cam_url):
        self.cap = cv2.VideoCapture(cam_url)
        self.frontal = cv2.CascadeClassifier('models/frontalface.xml')
        self.profile = cv2.CascadeClassifier('models/profileface.xml')
        self.body = cv2.CascadeClassifier('models/fullbody.xml')
        Thread.__init__(self)

    def run(self):
        ret, frame = self.cap.read() if self.cap.isOpened() else False, False
        if ret:
            while True:
                ret, frame = self.cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                detections = self.body.detectMultiScale(gray, 1.1, 5)
                if len(detections) == 0:
                    detections = self.frontal.detectMultiScale(gray, 1.1, 5)
                if len(detections) == 0:
                    detections = self.profile.detectMultiScale(gray, 1.1, 5)
                for (x, y, w, h) in detections:
                    # display the detected boxes in the colour picture
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                cv2.imshow("Camera", frame)  

                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # esc key ends process
                    self.cap.release()
                    break
            cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = ESP32SecurityCam("http://192.168.0.27:81/stream")
    cam.start()
