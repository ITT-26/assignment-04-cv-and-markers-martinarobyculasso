import argparse
import cv2
from PIL import Image
from pathlib import Path
import numpy as np

WINDOW_NAME = "Image Extractor"
MAX_POINTS = 4
POINT_COLOR = (255, 0, 0)  # BGR

points = []


# mouse callback -> handling clicks for area selection/transformation
def mouse_callback(event, x, y, flags, param):
    global img, points, img_transformed

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < MAX_POINTS:
            points.append([x, y])
            cv2.circle(img, (x, y), 5, POINT_COLOR, -1)
            cv2.imshow(WINDOW_NAME, img)

            # after the user has selected 4 points, begin with the setup for the transformation
            if len(points) == MAX_POINTS:

                print("\nPress 's' to save the transformed image.")

                # order the selected points (top-left, top-right, bottom-right, bottom-left)
                pts = np.float32(points)
                pts = pts[np.argsort(pts[:, 1])]
                top = pts[:2]
                bottom = pts[2:]
                top = top[np.argsort(top[:, 0])]
                bottom = bottom[np.argsort(bottom[:, 0])]

                # source points
                source = np.float32([top[0], top[1], bottom[1], bottom[0]])

                # destination points using specified res
                destination = np.float32(
                    [[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]]
                )

                # get transformation matrix
                mat = cv2.getPerspectiveTransform(source, destination)

                # apply transformation matrix, store the result in img_transformed
                img_transformed = cv2.warpPerspective(
                    img_original, mat, (out_w, out_h), flags=cv2.INTER_LINEAR
                )

                # show result
                cv2.imshow(WINDOW_NAME, img_transformed)

        else:
            print("Maximum number of points reached. Press [ESC] to discard changes.")


parser = argparse.ArgumentParser(description="Image Extractor")
parser.add_argument("input", help="Path to input image")
parser.add_argument("output", help="Path to save result")
parser.add_argument("width", type=int, help="Output width in pixels")
parser.add_argument("height", type=int, help="Output height in pixels")
args = parser.parse_args()

raw_path = Path(args.input)
save_path = Path(args.output)
out_w = args.width
out_h = args.height

# load image
img = cv2.imread(str(raw_path))
if img is None:
    print(f"Could not open image: {raw_path}")
    exit(1)
img_original = img.copy()

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
cv2.imshow(WINDOW_NAME, img)
print("\nClick 4 points to select the region. Press [ESC] to reset, [q] to quit.")

img_transformed = None

while True:
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

    key = cv2.waitKey(1) & 0xFF

    # save with 's' (only works with the result img)
    if key == ord("s"):
        if img_transformed is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            pil_img = Image.fromarray(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
            pil_img.save(str(save_path))
            print(f"\nSaved to {save_path}")
        else:
            print("No transformation yet. Select 4 points first.")

    # reset with ESC
    elif key == 27:
        points = []
        img_transformed = None
        img = img_original.copy()
        cv2.imshow(WINDOW_NAME, img)
        print("\nReset. Click 4 points to start over, [q] to quit.")

    # quit with 'q'
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
