from cam import ESP32SecurityCam

if __name__ == "__main__":
    cam_threads = []

    cam = ESP32SecurityCam("http://192.168.0.27:81/stream")
    cam.start()
    cam_threads.append(cam)
