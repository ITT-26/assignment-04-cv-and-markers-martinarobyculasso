import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
from PIL import Image
import sys

WINDOW_WIDTH = 0
WINDOW_HEIGHT = 0
MISS_THRESHOLD = 45

last_source = None
miss_count = 0
video_id = 0

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])


# converts OpenCV image to PIL image and then to pyglet texture
# https://gist.github.com/nkymut/1cb40ea6ae4de0cf9ded7332f1ca0d55
def cv2glet(img, fmt):
    """Assumes image is in BGR color space. Returns a pyimg object"""
    if fmt == "GRAY":
        rows, cols = img.shape
        channels = 1
    else:
        rows, cols, channels = img.shape

    raw_img = Image.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels * cols
    pyimg = pyglet.image.ImageData(
        width=cols,
        height=rows,
        fmt=fmt,
        data=raw_img,
        pitch=top_to_bottom_flag * bytes_per_row,
    )
    return pyimg


def detect_board(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        return None

    source = [None] * 4
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        if 0 <= marker_id <= 3:
            source[marker_id] = marker_corners[0].mean(axis=0)

    if any(pt is None for pt in source):
        return None

    # order points by position: TL, TR, BL, BR
    pts = np.float32(source)
    pts = pts[np.argsort(pts[:, 1])]
    top = pts[:2][np.argsort(pts[:2, 0])]
    bottom = pts[2:][np.argsort(pts[2:, 0])]

    return np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32)


def warp_frame(frame, source):
    global WINDOW_WIDTH, WINDOW_HEIGHT

    # destination points using camera res
    destination = np.float32(
        [
            [0, 0],
            [WINDOW_WIDTH, 0],
            [0, WINDOW_HEIGHT],
            [WINDOW_WIDTH, WINDOW_HEIGHT],
        ]
    )

    # get transformation matrix
    mat = cv2.getPerspectiveTransform(source, destination)

    # apply transformation matrix, store the result in warped
    warped = cv2.warpPerspective(
        frame, mat, (WINDOW_WIDTH, WINDOW_HEIGHT), flags=cv2.INTER_LINEAR
    )

    return warped


def detect_finger(warped):
    pass


def update(dt):
    pass


# create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)

# get camera resolution -> set window width and height
WINDOW_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
WINDOW_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(f"Resolution: {int(WINDOW_WIDTH)}x{int(WINDOW_HEIGHT)}")

# ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# create pyglet window
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)


@window.event
def on_key_press(key, modifiers):
    if key == pyglet.window.key.ESCAPE or key == pyglet.window.key.Q:
        pyglet.app.exit()

@window.event
def on_close():
    cap.release()
    pyglet.app.exit()

@window.event
def on_draw():
    global last_source, miss_count
    window.clear()
    ret, frame = cap.read()
    source = detect_board(frame)
    if source is not None:
        last_source = source
        miss_count = 0
    else:
        miss_count += 1
        if miss_count > MISS_THRESHOLD:
            last_source = None
    if last_source is not None:
        warped = warp_frame(frame, last_source)
        img = cv2glet(warped, "BGR")
    else:
        img = cv2glet(frame, "BGR")

    img.blit(0, 0, 0)


pyglet.clock.schedule_interval(update, 1 / 60)
pyglet.app.run()
cap.release()
