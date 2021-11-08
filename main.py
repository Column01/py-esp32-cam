from cam import ESP32SecurityCam
from flask import Flask, Response, render_template

app = Flask("Camera Feeds")
cam_threads = []


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/stream")
def stream():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen():
    while True:
        #get camera frame
        frame = cam_threads[0].get_frame()
        if frame is None:
            raise Exception("The frame could not be found!")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


if __name__ == "__main__":
    cam = ESP32SecurityCam("http://192.168.0.27:81/stream")
    cam.start()
    cam_threads.append(cam)
    app.run("0.0.0.0", port=5000)
