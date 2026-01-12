"""VirtualMouse utilities — includes finger counting helper using HandDetector.

Function:
- countFingers(detector, img, handNo=0, draw=True)

Uses tip ids [4, 8, 12, 16, 20]. For index->pinky a finger is considered "up" when
its tip y is less than the PIP joint y (origin at top-left). Thumb is handled using
x-coordinates and handedness: for a right hand thumb is up when tip.x < ip.x, for a
left hand when tip.x > ip.x.
"""

import cv2
from typing import List, Tuple, Optional
from HandTrackingModule import HandDetector


def countFingers(detector: HandDetector, img, handNo: int = 0, draw: bool = True) -> Tuple[int, List[int]]:
    """Count how many fingers are up for a detected hand.

    Args:
        detector: HandDetector instance (should have processed the image via findHands).
        img: BGR image (OpenCV format).
        handNo: Index of the hand to evaluate.
        draw: Whether to draw the count text on the image.

    Returns:
        total: number of fingers up (0-5).
        fingers: list of binary values [thumb, index, middle, ring, pinky].
    """
    lmList, bbox = detector.findPosition(img, handNo=handNo, draw=draw)
    if not lmList:
        return 0, []

    # Create mapping id -> (x, y)
    lm = {item[0]: (item[1], item[2]) for item in lmList}
    tipIds = [4, 8, 12, 16, 20]

    fingers: List[int] = []

    # Try to get handedness info from detector.results
    handType: Optional[str] = None
    try:
        if detector.results and detector.results.multi_handedness:
            handType = detector.results.multi_handedness[handNo].classification[0].label
    except Exception:
        handType = None

    # Thumb: compare x-coordinates depending on hand side
    if 4 in lm and 3 in lm:
        if handType == "Right":
            fingers.append(1 if lm[4][0] < lm[3][0] else 0)
        else:
            fingers.append(1 if lm[4][0] > lm[3][0] else 0)
    else:
        fingers.append(0)

    # Other fingers: tip.y < pip.y indicates finger is up
    for tipId in tipIds[1:]:
        pipId = tipId - 2
        if tipId in lm and pipId in lm:
            fingers.append(1 if lm[tipId][1] < lm[pipId][1] else 0)
        else:
            fingers.append(0)

    total = sum(fingers)

    if draw and bbox is not None:
        x, y, w, h = bbox
        cv2.putText(img, f"Fingers: {total}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return total, fingers


def run_virtual_mouse(window_name: str = "VirtualMouse - Finger Counter") -> str:
    """Run the VirtualMouse finger counter demo.

    Returns:
        'back' to return to launcher, 'quit' to exit the application.
    """
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.7, trackCon=0.7)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        total, fingers = countFingers(detector, img)

        if fingers:
            # Optional console debug
            print("Fingers:", fingers, "Total:", total)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            # Go back to launcher
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return 'quit'

    cap.release()
    cv2.destroyWindow(window_name)
    return 'back'


if __name__ == "__main__":
    run_virtual_mouse()