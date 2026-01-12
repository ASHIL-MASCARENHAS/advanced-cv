"""Demo: Count how many fingers are up using HandDetector from utils.py.

- Function `countFingers(detector, img, handNo=0, draw=True)`:
  - Uses landmark IDs [4, 8, 12, 16, 20].
  - For fingers (index -> pinky) compares tip y with pip y (tip id - 2).
  - For thumb, uses handedness and compares x-coordinates between tip and preceding joint.
"""

import cv2
from HandTrackingModule import HandDetector
from typing import List, Tuple


def countFingers(detector: HandDetector, img, handNo: int = 0, draw: bool = True) -> Tuple[int, List[int]]:
    """Count how many fingers are up for the specified hand.

    Args:
        detector: Instance of `HandDetector` that has already processed the image (via `findHands`).
        img: BGR image (OpenCV format).
        handNo: Index of the hand to evaluate (0 = first detected).
        draw: Whether to draw the finger count text on the image.

    Returns:
        total (int): Number of fingers up (0-5).
        fingers (List[int]): Binary list for each finger [thumb, index, middle, ring, pinky].
    """
    lmList, bbox = detector.findPosition(img, handNo=handNo, draw=draw)
    if not lmList:
        return 0, []

    # Map landmarks to dict for easy access: id -> (x, y)
    lm = {item[0]: (item[1], item[2]) for item in lmList}
    tipIds = [4, 8, 12, 16, 20]

    fingers: List[int] = []

    # Determine handedness if available ("Left" / "Right")
    handType = None
    try:
        if detector.results and detector.results.multi_handedness:
            handType = detector.results.multi_handedness[handNo].classification[0].label
    except Exception:
        handType = None

    # Thumb: use x-coordinates depending on hand side
    # For 'Right' hand, thumb is considered up when tip.x < ip.x
    # For 'Left' hand, thumb is up when tip.x > ip.x
    if handType == "Right":
        fingers.append(1 if lm[4][0] < lm[3][0] else 0)
    else:
        fingers.append(1 if lm[4][0] > lm[3][0] else 0)

    # Other four fingers: tip.y < pip.y means finger is up (y origin is top-left)
    for tipId in tipIds[1:]:
        if lm[tipId][1] < lm[tipId - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    total = sum(fingers)

    if draw and bbox is not None:
        x, y, w, h = bbox
        cv2.putText(img, f"Fingers: {total}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return total, fingers


from VirtualMouse import run_virtual_mouse
from VirtualPainter import run_virtual_painter
from PoseModule import run_trainer


def launcher():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # Overlay the menu text
        cv2.putText(img, "Press 1 for Mouse, 2 for Painter, 3 for Trainer", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(img, "Press q to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Main Menu", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            # Launch Virtual Mouse mode
            cap.release()
            cv2.destroyWindow("Main Menu")
            res = run_virtual_mouse()
            if res == 'quit':
                break
            cap = cv2.VideoCapture(0)

        elif key == ord('2'):
            cap.release()
            cv2.destroyWindow("Main Menu")
            res = run_virtual_painter()
            if res == 'quit':
                break
            cap = cv2.VideoCapture(0)

        elif key == ord('3'):
            cap.release()
            cv2.destroyWindow("Main Menu")
            res = run_trainer()
            if res == 'quit':
                break
            cap = cv2.VideoCapture(0)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    launcher()