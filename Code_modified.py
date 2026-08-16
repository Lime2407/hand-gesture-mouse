import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import time
from collections import deque

# Settings
SMOOTHENING = 5
PINCH_THRESHOLD = 0.05
FIST_THRESHOLD = 0.15
SCROLL_SPEED = 2000
SCROLL_DEADZONE = 3
SCREEN_MARGIN = 50

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0
screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

MODEL_PATH = 'hand_landmarker.task'

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Colors
COLORS = {
    'palm_center': (255, 255, 255),
    'thumb': (255, 0, 0),
    'index': (0, 255, 255),
    'middle': (255, 255, 0),
    'ring': (0, 165, 255),
    'pinky': (0, 0, 255),
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

class PointSmoother:
    def __init__(self, smoothening=5):
        self.smoothening = smoothening
        self.prev_x = 0
        self.prev_y = 0
        self.initialized = False

    def smooth(self, new_x, new_y):
        if not self.initialized:
            self.prev_x = new_x
            self.prev_y = new_y
            self.initialized = True
            return new_x, new_y

        curr_x = self.prev_x + (new_x - self.prev_x) / self.smoothening
        curr_y = self.prev_y + (new_y - self.prev_y) / self.smoothening

        self.prev_x = curr_x
        self.prev_y = curr_y

        return int(curr_x), int(curr_y)

cursor_smoother = PointSmoother(smoothening=SMOOTHENING)

# States
left_button_pressed = False
right_click_cooldown = 0
last_scroll_y = 0

def calculate_distance(point1, point2):
    return ((point1.x - point2.x)**2 + (point1.y - point2.y)**2)**0.5

def get_palm_center(hand_landmarks):
    points = [
        hand_landmarks[0], hand_landmarks[1], hand_landmarks[2], hand_landmarks[3], hand_landmarks[4],
        hand_landmarks[5], hand_landmarks[6], hand_landmarks[7], hand_landmarks[8],
        hand_landmarks[9], hand_landmarks[10], hand_landmarks[11], hand_landmarks[12],
        hand_landmarks[13], hand_landmarks[14], hand_landmarks[15], hand_landmarks[16],
        hand_landmarks[17], hand_landmarks[18], hand_landmarks[19], hand_landmarks[20]
    ]

    avg_x = sum(p.x for p in points) / len(points)
    avg_y = sum(p.y for p in points) / len(points)

    return type('Point', (), {'x': avg_x, 'y': avg_y})()

def is_pinch(hand_landmarks):
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    distance = calculate_distance(thumb_tip, index_tip)
    return distance < PINCH_THRESHOLD

def is_fist(hand_landmarks):
    """Defines the fist"""
    fingers = [8, 12, 16, 20]
    bases = [6, 10, 14, 18]

    for tip, base in zip(fingers, bases):
        if hand_landmarks[tip].y < hand_landmarks[base].y:
            return False

    thumb_tip = hand_landmarks[4]
    wrist = hand_landmarks[0]
    thumb_distance = calculate_distance(thumb_tip, wrist)

    return thumb_distance < FIST_THRESHOLD

def is_peace_sign(hand_landmarks):
    """Detects a peace sign: index and middle fingers up, ring and pinky down."""
    index_up = hand_landmarks[8].y < hand_landmarks[6].y
    middle_up = hand_landmarks[12].y < hand_landmarks[10].y
    ring_down = hand_landmarks[16].y > hand_landmarks[14].y
    pinky_down = hand_landmarks[20].y > hand_landmarks[18].y
    return index_up and middle_up and ring_down and pinky_down


def is_scroll_gesture(hand_landmarks):
    """
    Scroll Gesture: pointing and pinky raised
    """
    index_tip = hand_landmarks[8].y
    index_pip = hand_landmarks[6].y
    index_up = index_tip < index_pip

    pinky_tip = hand_landmarks[20].y
    pinky_pip = hand_landmarks[18].y
    pinky_up = pinky_tip < pinky_pip

    middle_tip = hand_landmarks[12].y
    middle_pip = hand_landmarks[10].y
    middle_down = middle_tip > middle_pip

    ring_tip = hand_landmarks[16].y
    ring_pip = hand_landmarks[14].y
    ring_down = ring_tip > ring_pip

    return index_up and pinky_up and middle_down and ring_down

def map_to_screen_coordinates(x, y, frame_width, frame_height):
    screen_x = np.interp(x, [SCREEN_MARGIN, frame_width - SCREEN_MARGIN],
                        [0, screen_width])
    screen_y = np.interp(y, [SCREEN_MARGIN, frame_height - SCREEN_MARGIN],
                        [0, screen_height])

    screen_x = max(0, min(screen_width, screen_x))
    screen_y = max(0, min(screen_height, screen_y))

    return screen_x, screen_y

# FPS tracking
prev_time = 0
fps_history = deque(maxlen=30)

print("Programm activated!")
print("Control:")
print("- LEFT hand:")
print("  * Opened palm -> moving cursor")
print("  * Pinch (thumb + pointing) -> LKM")
print("  * Gesture 'Goat' (pointing + pinkie) -> Scroll")
print("  * fist -> FREEZE THE CURSOR")
print("- RIGHT hand:")
print("  * fist -> right click")
print("ESC - exit")

while True:
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    display_image = image.copy()
    h, w, _ = display_image.shape

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imageRGB)

    detection_result = detector.detect(mp_image)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    fps_history.append(fps)
    avg_fps = int(np.mean(fps_history))

    cv2.putText(display_image, f"FPS: {avg_fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # State flags
    scroll_active = False
    left_hand_freeze = False
    left_hand_palm_center = None

    # Сначала определяем, есть ли кулак на левой руке
    if detection_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            handedness = "Unknown"
            if detection_result.handedness and idx < len(detection_result.handedness):
                handedness = detection_result.handedness[idx][0].category_name

            if handedness == "Left":
                left_hand_freeze = is_peace_sign(hand_landmarks)
                left_hand_palm_center = get_palm_center(hand_landmarks)
                break

    # Если левая рука в кулаке - ЗАМОРАЖИВАЕМ КУРСОР
    if left_hand_freeze:
        cv2.putText(display_image, "CURSOR FROZEN - PEACE SIGN", (w//2 - 200, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Process all hands
    if detection_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            handedness = "Unknown"
            if detection_result.handedness and idx < len(detection_result.handedness):
                handedness = detection_result.handedness[idx][0].category_name

            palm_center = get_palm_center(hand_landmarks)
            palm_x = int(palm_center.x * w)
            palm_y = int(palm_center.y * h)

            thumb = hand_landmarks[4]
            index = hand_landmarks[8]
            middle = hand_landmarks[12]
            ring = hand_landmarks[16]
            pinky = hand_landmarks[20]

            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
            index_x, index_y = int(index.x * w), int(index.y * h)
            middle_x, middle_y = int(middle.x * w), int(middle.y * h)
            ring_x, ring_y = int(ring.x * w), int(ring.y * h)
            pinky_x, pinky_y = int(pinky.x * w), int(pinky.y * h)

            # Detect gestures
            fist = is_fist(hand_landmarks)

            if handedness == "Left":
                pinch = is_pinch(hand_landmarks)
                scroll = is_scroll_gesture(hand_landmarks)

                # Draw hand skeleton
                line_color = (0, 255, 0) if not fist else (100, 100, 100)
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_point = hand_landmarks[start_idx]
                    end_point = hand_landmarks[end_idx]

                    start_x = int(start_point.x * w)
                    start_y = int(start_point.y * h)
                    end_x = int(end_point.x * w)
                    end_y = int(end_point.y * h)

                    cv2.line(display_image, (start_x, start_y), (end_x, end_y),
                            line_color, 1)

                # Active zone
                cv2.rectangle(display_image,
                             (SCREEN_MARGIN, SCREEN_MARGIN),
                             (w - SCREEN_MARGIN, h - SCREEN_MARGIN),
                             (255, 255, 255), 1)

                # Управление курсором ТОЛЬКО если левая рука НЕ в кулаке
                if not left_hand_freeze:
                    if (SCREEN_MARGIN < palm_x < w - SCREEN_MARGIN and
                        SCREEN_MARGIN < palm_y < h - SCREEN_MARGIN):

                        # Move cursor
                        screen_x, screen_y = map_to_screen_coordinates(palm_x, palm_y, w, h)
                        smooth_x, smooth_y = cursor_smoother.smooth(screen_x, screen_y)
                        pyautogui.moveTo(smooth_x, smooth_y)

                        # LMB (pinch)
                        if pinch and not left_button_pressed:
                            pyautogui.mouseDown(button='left')
                            left_button_pressed = True
                            print("LMB pressed")
                        elif not pinch and left_button_pressed:
                            pyautogui.mouseUp(button='left')
                            left_button_pressed = False
                            print("LMB released")

                        # SCROLL (goat gesture)
                        if scroll:
                            scroll_active = True
                            cv2.putText(display_image, "SCROLL MODE", (w//2 - 70, 100),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                            if last_scroll_y != 0:
                                delta = (index_y - last_scroll_y) * 2.0
                                if abs(delta) > SCROLL_DEADZONE:
                                    scroll_amount = int(-delta * SCROLL_SPEED / 60)
                                    pyautogui.scroll(scroll_amount)

                                    dir_text = "▲ UP" if delta < 0 else "▼ DOWN"
                                    cv2.putText(display_image, dir_text, (w//2 - 30, 140),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                            last_scroll_y = index_y

                # Визуализация для левой руки
                hand_status = "PEACE (FROZEN)" if is_peace_sign(hand_landmarks) else ("FIST" if fist else "ACTIVE")
                hand_color = (100, 100, 100) if fist else (0, 255, 0)
                cv2.putText(display_image, f"LEFT - {hand_status}", (palm_x - 50, palm_y - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2)

                if is_peace_sign(hand_landmarks):
                    cv2.circle(display_image, (palm_x, palm_y), 40, (100, 100, 100), 3)
                    cv2.putText(display_image, "FROZEN", (palm_x - 30, palm_y - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                else:
                    if pinch:
                        cv2.line(display_image, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 5)
                        cv2.putText(display_image, "LMB", (thumb_x - 20, thumb_y - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if scroll:
                        cv2.circle(display_image, (index_x, index_y), 25, (255, 255, 0), 3)
                        cv2.circle(display_image, (pinky_x, pinky_y), 25, (255, 255, 0), 3)
                        cv2.putText(display_image, "🤘", (index_x - 20, index_y - 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)

            elif handedness == "Right":
                # ПРАВАЯ РУКА: правый клик работает всегда
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_point = hand_landmarks[start_idx]
                    end_point = hand_landmarks[end_idx]

                    start_x = int(start_point.x * w)
                    start_y = int(start_point.y * h)
                    end_x = int(end_point.x * w)
                    end_y = int(end_point.y * h)

                    cv2.line(display_image, (start_x, start_y), (end_x, end_y),
                            (0, 0, 255), 1)

                # Правый клик
                if fist and (curr_time - right_click_cooldown) > 0.5:
                    pyautogui.click(button='right')
                    right_click_cooldown = curr_time
                    cv2.putText(display_image, "RIGHT CLICK!", (w//2 - 80, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                    print("Right click")

                cv2.putText(display_image, "RIGHT", (palm_x - 30, palm_y - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if fist:
                    cv2.circle(display_image, (palm_x, palm_y), 40, (0, 0, 255), 3)
                    cv2.putText(display_image, "FIST", (palm_x - 20, palm_y - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # General visualization
            cv2.circle(display_image, (palm_x, palm_y), 15, (0, 0, 0), -1)
            cv2.circle(display_image, (palm_x, palm_y), 13, (255, 255, 255), -1)
            cv2.circle(display_image, (palm_x, palm_y), 15, (255, 255, 255), 2)

            cv2.circle(display_image, (thumb_x, thumb_y), 8, (255, 0, 0), -1)
            cv2.circle(display_image, (index_x, index_y), 8, (0, 255, 255), -1)
            cv2.circle(display_image, (middle_x, middle_y), 8, (255, 255, 0), -1)
            cv2.circle(display_image, (ring_x, ring_y), 8, (0, 165, 255), -1)
            cv2.circle(display_image, (pinky_x, pinky_y), 8, (0, 0, 255), -1)

        # Reset scroll
        if not scroll_active:
            last_scroll_y = 0

        # Status panel
        status_y = 60
        cv2.putText(display_image, f"Left button: {'PRESSED' if left_button_pressed else 'up'}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 255, 0) if left_button_pressed else (100, 100, 100), 2)

        cursor_status = "FROZEN" if left_hand_freeze else "ACTIVE"
        cursor_color = (0, 0, 255) if left_hand_freeze else (0, 255, 0)
        cv2.putText(display_image, f"Cursor: {cursor_status}",
                   (10, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   cursor_color, 2)

        cv2.putText(display_image, f"Scroll: {'ACTIVE' if scroll_active else 'inactive'}",
                   (10, status_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 0) if scroll_active else (100, 100, 100), 2)

    else:
        cv2.putText(display_image, "No hands detected", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Instructions
    cv2.putText(display_image, "LEFT: open=move, pinch=LMB, fist=RMB, 🤘=scroll, PEACE=FREEZE", (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(display_image, "RIGHT: visualization only", (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(display_image, "ESC: Exit", (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    cv2.imshow("Hand Control", display_image)

    if cv2.waitKey(1) & 0xFF == 27:

        break
cap.release()
cv2.destroyAllWindows()
print("Program finished")