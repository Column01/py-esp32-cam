from cam import ESP32SecurityCam, FrameSize
from flask import Flask, Response, render_template

app = Flask("Camera Feeds")
cam_threads = []


@app.route("/")
def index():
    return render_template('index.jinja', cam_threads=cam_threads)


@app.route("/stream/<index>")
def stream(index):
    return Response(gen(index), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen(index):
    if index == None:
        index = 0
    index = int(index)
    while True:
        # Get camera frame
        frame = cam_threads[index].get_frame()
        if frame is None:
            raise Exception("The frame could not be found!")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


if __name__ == "__main__":
    cam = ESP32SecurityCam("http://192.168.0.27", FrameSize.FRAMESIZE_SVGA, (True, False), 100, 50, True)
    cam.start()
    cam_threads.append(cam)
    app.run("0.0.0.0", port=5000)
