"""Virtual Painter — draw on a canvas using hand gestures detected by MediaPipe.

Controls:
- Selection Mode: Index and Middle fingers up → move cursor without drawing.
  - If cursor hits the header area (top bar), select a color/eraser.
- Drawing Mode: Only Index finger up → draw on a separate canvas.

Usage:
    python VirtualPainter.py

Press 'q' to quit.
"""

import cv2
import numpy as np
from HandTrackingModule import HandDetector
from typing import List, Tuple


def fingersUpFromLm(lmList: List[List[int]], results, handNo: int = 0) -> List[int]:
    """Return list of 5 binary values for fingers [thumb, index, middle, ring, pinky].

    Uses tip ids [4, 8, 12, 16, 20]. Index/middle determination uses y-coordinates
    (tip y < pip y means finger up). Thumb uses handedness and x-coordinates.
    """
    tips = [4, 8, 12, 16, 20]
    fingers = [0, 0, 0, 0, 0]

    if not lmList:
        return fingers

    lm_dict = {id: (x, y) for id, x, y in lmList}

    # Get hand type if available
    handType = None
    try:
        if results and results.multi_handedness:
            handType = results.multi_handedness[handNo].classification[0].label
    except Exception:
        handType = None

    # Thumb
    if 4 in lm_dict and 3 in lm_dict:
        if handType == "Right":
            fingers[0] = 1 if lm_dict[4][0] < lm_dict[3][0] else 0
        else:
            fingers[0] = 1 if lm_dict[4][0] > lm_dict[3][0] else 0

    # Other fingers: tip.y < pip.y (y increases downward)
    for i, tipId in enumerate(tips[1:], start=1):
        pipId = tipId - 2
        if tipId in lm_dict and pipId in lm_dict:
            fingers[i] = 1 if lm_dict[tipId][1] < lm_dict[pipId][1] else 0

    return fingers


def run_virtual_painter(window_name: str = "Virtual Painter") -> str:
    """Run the virtual painter demo. Returns 'back' to return to launcher or 'quit' to exit."""
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(detectionCon=0.85, trackCon=0.75)

    # Canvas for drawing
    imgCanvas = None

    # Header (color selection) config
    header_height = 125
    colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0)]  # magenta, blue, green, eraser(black)
    color_names = ["Magenta", "Blue", "Green", "Eraser"]
    current_color = colors[0]
    brush_thickness = 15
    eraser_thickness = 50

    xp, yp = 0, 0

    while True:
        success, img = cap.read()
        if not success:
            break

        # Flip for mirror view
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        if imgCanvas is None:
            imgCanvas = np.zeros_like(img)

        # Draw header UI
        num_buttons = len(colors)
        btn_width = w // num_buttons
        for i, color in enumerate(colors):
            x1 = i * btn_width
            x2 = (i + 1) * btn_width
            cv2.rectangle(img, (x1, 0), (x2, header_height), color, -1)
            cv2.putText(img, color_names[i], (x1 + 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255) if color != (0, 0, 0) else (255, 255, 255), 2)

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if lmList:
            fingers = fingersUpFromLm(lmList, detector.results, handNo=0)

            # Get coordinates of index and middle finger tips
            _, ix, iy = lmList[8]
            _, mx, my = lmList[12]

            # Selection mode: index + middle up
            if fingers[1] == 1 and fingers[2] == 1:
                xp, yp = 0, 0
                cv2.circle(img, (ix, iy), 15, (0, 0, 255), cv2.FILLED)

                # If in header area, choose color
                if iy < header_height:
                    # Determine which button
                    btn_idx = ix // btn_width
                    if 0 <= btn_idx < num_buttons:
                        current_color = colors[btn_idx]
                        # Visual feedback on top bar
                        x1 = btn_idx * btn_width
                        x2 = (btn_idx + 1) * btn_width
                        cv2.rectangle(img, (x1, 0), (x2, header_height), current_color, -1)
                        cv2.putText(img, f"Selected: {color_names[btn_idx]}", (10, header_height + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Drawing mode: only index up
            elif fingers[1] == 1 and fingers[2] == 0:
                cv2.circle(img, (ix, iy), 10, current_color, cv2.FILLED)

                if xp == 0 and yp == 0:
                    xp, yp = ix, iy

                # Eraser
                if current_color == (0, 0, 0):
                    cv2.line(imgCanvas, (xp, yp), (ix, iy), current_color, eraser_thickness)
                else:
                    cv2.line(imgCanvas, (xp, yp), (ix, iy), current_color, brush_thickness)

                xp, yp = ix, iy

            else:
                xp, yp = 0, 0

        # Merge canvas and frame
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        # Show instructions
        cv2.putText(img, "Selection: Index+Middle  |  Draw: Index  |  Press 'b' to go back", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window_name, img)
        # cv2.imshow("Canvas", imgCanvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return 'quit'

    cap.release()
    cv2.destroyWindow(window_name)
    return 'back'


if __name__ == "__main__":
    run_virtual_painter()