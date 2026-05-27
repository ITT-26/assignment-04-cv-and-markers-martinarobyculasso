[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/5NorvP5a)

#  Markers, Computer Vision, and AR 

Assignment 4 for the Interactive Techniques and Technologies course (ITT), Universität Regensburg.

Author: Martina Roby Culasso

---

Each folder contains an `info.txt` file with a description of the files and relevant notes.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Exercise 1 - Perspective Transformation

A command-line tool that loads an image and lets the user select 4 corner points by clicking. The selected region is perspectively warped to a rectangle at the specified output resolution and displayed. The result can be saved or the selection reset at any time.

> An example image is included in the img_raw/ folder.

**Usage:**
```bash
cd perspective_transformation
python image_extractor.py <input> <output> <width> <height>  
```
| Argument | Description |
|-----|--------|
| `input` | Path to the input image |
| `output` | Path to save the result |
| `width` | 	Output width in pixels |
| `height` | 	Output height in pixels |

Example: ``python image_extractor.py img_raw/sample_image.jpg img_transf/result.jpg 800 600``

**Controls:**
| Key | Action |
|-----|--------|
| Click | Select corner point (4 required) |
| `S` | Save the warped result |
| `ESC` | Reset selection |
| `Q` | Quit |


## Exercise 2 - AR Game

An augmented reality sorting game using a physical board with ArUco markers at each corner. The webcam feed is warped to fill the window, and finger tracking via skin color detection allows players to drag animal sprites into their matching colored zones.

> Requires a webcam and a physical board with four ArUco markers at the corners.

**Usage:**
```bash
cd ar_game
python AR_game.py
```
**How to play:**
- Place the ArUco board in front of the camera to start the game.
- Drag **pigs** to the **pink zone** (top).
- Drag **hippos** to the **blue zone** (bottom).
- Placing an animal in the wrong zone sends it back to its starting position.
- Sort all animals as fast as possible!

**Controls:**
| Key | Action |
|-----|--------|
| `R` | Restart (from results screen) |
| `Q` / `ESC` | Quit |

