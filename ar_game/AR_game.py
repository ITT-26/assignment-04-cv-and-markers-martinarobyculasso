import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
from PIL import Image
import sys
import os
import random

WINDOW_WIDTH = 0
WINDOW_HEIGHT = 0
MISS_THRESHOLD = 45  # max number of frames in which 4 markers are not detected
N = 4  # number of sprites for each animal

last_source = None
miss_count = 0
finger_box = None  # current fingertip collision box in pyglet coordinates
score = 0
elapsed_time = 0.0  # seconds elapsed during playing state
video_id = 0

# colors
PINK = (220, 107, 173)
BLUE = (113, 146, 190)
WHITE = (228, 223, 218)
PURPLE = (140, 122, 169)
BLACK = (0, 0, 0)

game_states = ["instructions", "playing", "results"]
state = "instructions"

# path for assets
assets_dir = os.path.join(os.path.dirname(__file__), "assets")
pyglet.resource.path = [assets_dir]
pyglet.resource.reindex()

# font
pyglet.font.add_file(os.path.join(assets_dir, "Fredoka-VariableFont_wdth,wght.ttf"))

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

    # only return source array if 4 markers are visible
    if ids is None or len(ids) < 4:
        return None

    source = [None] * 4
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        if 0 <= marker_id <= 3:
            source[marker_id] = marker_corners[0].mean(axis=0)

    if any(pt is None for pt in source):
        return None

    # order points by position (TL, TR, BL, BR)
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

    # blur
    warped = cv2.GaussianBlur(warped, (5, 5), 0)

    # BGR -> HSV
    hsv_image = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    # skin color range in HSV (set up by testing)
    lower_skin = np.array([0, 30, 60])
    upper_skin = np.array([20, 170, 255])

    # create mask for skin
    # NOTE: tried detecting white background first and inverting the mask,
    # but it wasn't very reliable because of shadows
    mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # clean up mask using morph. operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if no contours, return none
    if len(contours) == 0:
        return warped, None

    # find largest contour -> assume it is the hand/finger
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(warped, [largest], -1, (0, 0, 255), 4)

    # find the topmost point of the contour (should be tip of the finger)
    topmost = tuple(largest[largest[:, :, 1].argmin()][0])

    # lower the reference point a little bit
    centroid = (topmost[0], topmost[1] + 10)

    # define a small collision box around the fingertip (to be used for collisions with game sprites)
    box_size = 30
    x1 = centroid[0] - box_size
    y1 = centroid[1] - box_size
    x2 = centroid[0] + box_size
    y2 = centroid[1] + box_size

    # draw the collision box for visualization
    cv2.rectangle(warped, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return warped, (x1, y1, x2, y2)


class AnimalSprite:
    # load images once
    pig_img = pyglet.resource.image("pig.png")
    hippo_img = pyglet.resource.image("hippo.png")

    def __init__(self, animal_type, x, y):
        self.type = animal_type  # "pig" or "hippo"

        # pick the correct image based on type
        img = self.pig_img if animal_type == "pig" else self.hippo_img

        # center the image anchor so position refers to the center of the sprite
        img.anchor_x = img.width // 2
        img.anchor_y = img.height // 2

        # create pyglet sprite
        self.sprite = pyglet.sprite.Sprite(img, x=x, y=y)
        self.sprite.scale = 0.15  # adjust size to fit board

        # initial spawn position
        self.home_x = x
        self.home_y = y

        self.held = False  # finger currently grabbing this sprite ?
        self.sorted = False  # sprite placed in its zone ?

    def draw(self):
        if not self.sorted:
            self.sprite.draw()

    def update_position(self, x, y):
        self.sprite.x = x
        self.sprite.y = y

    # return to starting position if taken into the wrong area
    def bounce_back(self):
        self.sprite.x = self.home_x
        self.sprite.y = self.home_y
        self.held = False

    # get sprite bounding box for collisions
    def get_rect(self):
        hw = self.sprite.width // 2
        hh = self.sprite.height // 2
        return (
            self.sprite.x - hw,
            self.sprite.y - hh,
            self.sprite.x + hw,
            self.sprite.y + hh,
        )


def spawn_sprites():
    sprites = []
    placed_positions = []

    # safe spawn area: between the two zones, with some padding between sprites
    padding = 60
    min_distance = 100  # minimum pixels between sprite centers
    x_min = padding
    x_max = WINDOW_WIDTH - padding
    y_min = ZONE_HEIGHT + padding
    y_max = WINDOW_HEIGHT - ZONE_HEIGHT - padding

    for animal_type in ["pig"] * N + ["hippo"] * N:
        # try up to 50 times to find a non-overlapping position
        for _ in range(50):
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            # check distance from all already placed sprites
            too_close = any(
                ((x - px) ** 2 + (y - py) ** 2) ** 0.5 < min_distance
                for px, py in placed_positions
            )
            if not too_close:
                break
        placed_positions.append((x, y))
        sprites.append(AnimalSprite(animal_type, x, y))

    return sprites


# active sprites in the game
sprites = []


# check if two rectangles collide
def rects_overlap(r1, r2):
    return r1[0] < r2[2] and r1[2] > r2[0] and r1[1] < r2[3] and r1[3] > r2[1]


# check if sprite center point is inside colored rectangle zone
def in_zone(sprite, zone):
    cx = sprite.sprite.x
    cy = sprite.sprite.y
    return zone.x <= cx <= zone.x + zone.width and zone.y <= cy <= zone.y + zone.height


def update(dt):
    global finger_box, score, state, elapsed_time

    if state != "playing":
        return

    # increment timer
    elapsed_time += dt

    if finger_box is None:
        return

    # convert finger_box from OpenCV coords (y=0 top) to pyglet coords (y=0 bottom)
    fx1, fy1, fx2, fy2 = finger_box
    pyglet_finger_box = (fx1, WINDOW_HEIGHT - fy2, fx2, WINDOW_HEIGHT - fy1)

    # finger center in pyglet coords
    fcx = (pyglet_finger_box[0] + pyglet_finger_box[2]) // 2
    fcy = (pyglet_finger_box[1] + pyglet_finger_box[3]) // 2

    for sprite in sprites:
        if sprite.sorted:
            continue

        if sprite.held:
            # move sprite to finger center
            sprite.update_position(fcx, fcy)

            # check if sprite is in the correct zone
            correct_zone = zone_pink if sprite.type == "pig" else zone_blue
            wrong_zone = zone_blue if sprite.type == "pig" else zone_pink

            if in_zone(sprite, correct_zone):
                sprite.sorted = True
                sprite.held = False
                score += 1
                # check if all sprites are sorted -> game ends
                if all(s.sorted for s in sprites):
                    state = "results"

            elif in_zone(sprite, wrong_zone):
                # bounce back to home position
                sprite.bounce_back()

        else:
            # check if finger overlaps this sprite and grab it
            # only grab one sprite at a time
            already_holding = any(s.held for s in sprites)
            if not already_holding and rects_overlap(
                pyglet_finger_box, sprite.get_rect()
            ):
                sprite.held = True


# create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)

# get camera resolution ->  window width and height
WINDOW_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
WINDOW_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# create pyglet window
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

# GAME ZONES
# zone height: 20% of the window
ZONE_HEIGHT = WINDOW_HEIGHT // 5

# pink zone at the top, blue zone at the bottom
zone_pink = pyglet.shapes.Rectangle(
    x=0,
    y=WINDOW_HEIGHT - ZONE_HEIGHT,
    width=WINDOW_WIDTH,
    height=ZONE_HEIGHT,
    color=PINK,
)
zone_blue = pyglet.shapes.Rectangle(
    x=0,
    y=0,
    width=WINDOW_WIDTH,
    height=ZONE_HEIGHT,
    color=BLUE,
)
# transparency
zone_pink.opacity = 120
zone_blue.opacity = 120

# LABELS

# instructions screen
instructions_label = pyglet.text.Label(
    "Sort the animals with your finger!\n\nPigs go to the PINK zone\nHippos go to the BLUE zone\n\nPlace the board in front of the camera to start",
    font_name="Fredoka",
    font_size=20,
    color=WHITE + (255,),
    x=WINDOW_WIDTH // 2,
    y=WINDOW_HEIGHT // 2,
    anchor_x="center",
    anchor_y="center",
    multiline=True,
    width=500,
    align="center",
)

# score label (top-left during playing)
score_label = pyglet.text.Label(
    "Score: 0",
    font_name="Fredoka",
    font_size=20,
    color=BLACK + (255,),
    x=10,
    y=WINDOW_HEIGHT - 10,
    anchor_x="left",
    anchor_y="top",
)

# timer label (top-right during playing)
timer_label = pyglet.text.Label(
    "Time: 0.0s",
    font_name="Fredoka",
    font_size=20,
    color=BLACK + (255,),
    x=WINDOW_WIDTH - 10,
    y=WINDOW_HEIGHT - 10,
    anchor_x="right",
    anchor_y="top",
)

# results screen
results_label = pyglet.text.Label(
    "",
    font_name="Fredoka",
    font_size=24,
    color=WHITE + (255,),
    x=WINDOW_WIDTH // 2,
    y=WINDOW_HEIGHT // 2,
    anchor_x="center",
    anchor_y="center",
    multiline=True,
    width=500,
    align="center",
)

# semi-transparent background for labels (instructions and results)
label_bg = pyglet.shapes.Rectangle(
    x=WINDOW_WIDTH // 2 - 280,
    y=WINDOW_HEIGHT // 2 - 120,
    width=560,
    height=240,
    color=PURPLE,
)
label_bg.opacity = 160


@window.event
def on_key_press(key, modifiers):
    global score, elapsed_time, sprites, state
    # close window with [ESC] or 'q'
    if key == pyglet.window.key.ESCAPE or key == pyglet.window.key.Q:
        pyglet.app.exit()
    # restart game with 'r' (only from results screen)
    elif key == pyglet.window.key.R and state == "results":
        score = 0
        elapsed_time = 0.0
        sprites = spawn_sprites()
        state = "playing"


@window.event
def on_close():
    cap.release()
    pyglet.app.exit()


@window.event
def on_draw():
    global last_source, miss_count, state, sprites, finger_box
    window.clear()
    ret, frame = cap.read()
    source = detect_board(frame)
    if source is not None:
        last_source = source
        miss_count = 0
        # first board detection -> game state transition
        if state == "instructions":
            state = "playing"
            sprites = spawn_sprites()
    else:
        miss_count += 1
        if miss_count > MISS_THRESHOLD:
            last_source = None
    if last_source is not None:
        warped = warp_frame(frame, last_source)
        warped, finger_box = detect_finger(warped)  # updates global finger_box
        img = cv2glet(warped, "BGR")
    else:
        img = cv2glet(frame, "BGR")
    img.blit(0, 0, 0)

    if state == "instructions":
        label_bg.draw()
        instructions_label.draw()

    elif state == "playing" and last_source is None:
        # board lost during game -> show prompt to replace it
        instructions_label.text = "Place the board in front\nof the camera to continue"
        label_bg.draw()
        instructions_label.draw()

    elif state == "playing" and last_source is not None:
        zone_pink.draw()
        zone_blue.draw()
        for sprite in sprites:
            sprite.draw()
        # update and draw score and timer
        score_label.text = f"Score: {score}"
        timer_label.text = f"Time: {elapsed_time:.1f}s"
        score_label.draw()
        timer_label.draw()

    elif state == "results":
        results_label.text = f"Well done!\n\nTime: {elapsed_time:.1f}s\n\nPress R to play again \nor Q/ESC to quit"
        label_bg.draw()
        results_label.draw()


pyglet.clock.schedule_interval(update, 1 / 60)
pyglet.app.run()
cap.release()
