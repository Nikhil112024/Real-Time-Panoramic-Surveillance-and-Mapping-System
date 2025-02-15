import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker

# Global dimensions for layout
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
BUTTON_HEIGHT = 100
LEFT_PANEL_HEIGHT = VIDEO_HEIGHT + BUTTON_HEIGHT
MAP_WIDTH = 640
MAP_HEIGHT = LEFT_PANEL_HEIGHT  # so both panels have the same height

# Initialize variables
selected_track_id = None  # ID of the selected person (only applicable for persons)
path_history = []         # History of centroids for the selected person
click_position = None     # Position of the mouse click
select_mode = False       # Whether we are in select mode

# -------------------------- DETECTION MODE SETUP --------------------------
# detection_mode:
#   0 -> People (only persons will be detected & tracked)
#   1 -> Object (only non-person objects are detected, no tracking IDs)
#   2 -> Both (persons are tracked, objects are only detected & labeled)
detection_mode = 0  # Start with "People" mode

# Initialize YOLO model and tracker
model = YOLO("yolov8n.pt")
tracker = Tracker()

# Retrieve class names from the YOLO model (if available)
class_names = model.model.names if hasattr(model.model, "names") else {}

# Camera index initialization
current_cam_index = 0  # start with the first camera
cap = cv2.VideoCapture(current_cam_index)
if not cap.isOpened():
    print(f"Error: Could not open camera with index {current_cam_index}.")

# Create canvases for the map and buttons
map_canvas = np.zeros((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)
button_canvas = np.zeros((BUTTON_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)

# -------------------------- BUTTONS SETUP --------------------------
# 1) Track button (for persons)
track_button_rect = (10, 30, 100, 70)
# 2) Cam button (switch camera)
cam_button_rect = (120, 30, 180, 70)
# 3) Mode button (people/object/both)
mode_button_rect = (200, 30, 290, 70)

# Grid dimensions for the map
num_rows = 6
num_cols = 8
row_height = map_canvas.shape[0] // num_rows
col_width = map_canvas.shape[1] // num_cols


def get_color(track_id, selected_track_id):
    """Assign red for the selected track ID, otherwise generate a unique color."""
    if track_id == selected_track_id:
        return (0, 0, 255)  # Red for the selected person
    np.random.seed(track_id)
    return tuple(int(c) for c in np.random.randint(0, 255, 3))


def draw_buttons():
    """
    Draw three buttons on the button canvas:
    1) Track: green if not tracking, red if tracking (only applies in People/Both mode)
    2) Cam: camera switch
    3) Mode: cycles People/Object/Both with color changes
    """
    button_canvas.fill(0)

    # -------------------------- TRACK BUTTON --------------------------
    # Only meaningful in People (0) or Both (2) modes.
    if detection_mode in (0, 2):
        if selected_track_id is None:
            track_color = (0, 255, 0)  # Green
        else:
            track_color = (0, 0, 255)  # Red
        track_text = "Track"
    else:
        # In Object mode, disable track button
        track_color = (50, 50, 50)
        track_text = "Track"

    x1, y1, x2, y2 = track_button_rect
    cv2.rectangle(button_canvas, (x1, y1), (x2, y2), track_color, -1)
    cv2.putText(button_canvas, track_text, (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # -------------------------- CAM BUTTON --------------------------
    cam_color = (255, 0, 0)  # Blue
    x1, y1, x2, y2 = cam_button_rect
    cv2.rectangle(button_canvas, (x1, y1), (x2, y2), cam_color, -1)
    cv2.putText(button_canvas, "Cam", (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # -------------------------- MODE BUTTON --------------------------
    # detection_mode: 0->People (green), 1->Object (blue), 2->Both (yellow)
    mode_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255)]
    mode_texts = ["People", "Object", "Both"]

    current_color = mode_colors[detection_mode]
    current_text = mode_texts[detection_mode]

    x1, y1, x2, y2 = mode_button_rect
    cv2.rectangle(button_canvas, (x1, y1), (x2, y2), current_color, -1)
    cv2.putText(button_canvas, current_text, (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_grid(canvas, num_rows, num_cols, row_height, col_width):
    """Draw a grid overlay on the given canvas."""
    for i in range(1, num_rows):
        y = i * row_height
        cv2.line(canvas, (0, y), (canvas.shape[1], y), (255, 255, 255), 1)
    for j in range(1, num_cols):
        x = j * col_width
        cv2.line(canvas, (x, 0), (x, canvas.shape[0]), (255, 255, 255), 1)


def switch_camera():
    """Switch to the next available camera by cycling the index."""
    global current_cam_index, cap
    new_cam_index = current_cam_index + 1
    cap.release()
    cap = cv2.VideoCapture(new_cam_index)
    if not cap.isOpened():
        print(f"Camera index {new_cam_index} not available. Reverting to index 0.")
        current_cam_index = 0
        cap.release()
        cap = cv2.VideoCapture(current_cam_index)
    else:
        current_cam_index = new_cam_index


def click_event(event, x, y, flags, param):
    """
    Handle mouse clicks for:
    1) Tracking toggle (when clicking "Track" button or clicking on a person)
    2) Camera switch (click on "Cam")
    3) Mode cycle (click on "People/Object/Both" button)
    4) Selecting a bounding box in the video feed if in select_mode
    """
    global click_position, selected_track_id, path_history, select_mode, detection_mode

    # Process only if click is in the left panel (first VIDEO_WIDTH pixels)
    if x >= VIDEO_WIDTH:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        # If click occurs in the button area (below the video feed)
        if y >= VIDEO_HEIGHT:
            y_adjusted = y - VIDEO_HEIGHT

            # Check if the click is within the TRACK button (only active in People/Both modes)
            tx1, ty1, tx2, ty2 = track_button_rect
            if tx1 <= x <= tx2 and ty1 <= y_adjusted <= ty2:
                if detection_mode in (0, 2):
                    # Toggle selection: if a person is selected, deselect; otherwise, enable select mode
                    if selected_track_id is None:
                        select_mode = True
                    else:
                        selected_track_id = None
                        path_history = []
                        select_mode = False
                return

            # Check if the click is within the CAM button
            cx1, cy1, cx2, cy2 = cam_button_rect
            if cx1 <= x <= cx2 and cy1 <= y_adjusted <= cy2:
                switch_camera()
                return

            # Check if the click is within the MODE button
            mx1, my1, mx2, my2 = mode_button_rect
            if mx1 <= x <= mx2 and my1 <= y_adjusted <= my2:
                detection_mode = (detection_mode + 1) % 3
                # Clear any previous selection when mode changes
                selected_track_id = None
                path_history = []
                return
        else:
            # Click in the video feed area
            click_position = (x, y)


# Set up mouse callback on the combined interface window
cv2.namedWindow("Main Interface")
cv2.setMouseCallback("Main Interface", click_event)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Frame not grabbed; attempting to reopen the camera...")
        cap.release()
        cap = cv2.VideoCapture(current_cam_index)
        continue

    frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

    # Run YOLO detection on the frame
    results = model(frame)
    # Lists to hold detections: each detection is [x1, y1, x2, y2, score, class_id]
    detections_all = []
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_id = int(class_id)
        detections_all.append([x1, y1, x2, y2, score, class_id])

    # -------------------------- MODE-SPECIFIC PROCESSING --------------------------
    if detection_mode == 0:
        # People only: filter for persons (class_id == 0) and track them.
        detections = []
        for det in detections_all:
            if int(det[5]) == 0:
                detections.append(det[:5])
        # Update tracker with person detections
        tracker.update(frame, detections)
        # For drawing, we use the tracker results.
    elif detection_mode == 1:
        # Object only: filter out persons (keep only detections where class_id != 0).
        object_detections = [det for det in detections_all if int(det[5]) != 0]
        # No tracking for objects.
    else:
        # Both: separate detections into persons and objects.
        person_detections = []
        object_detections = []
        for det in detections_all:
            if int(det[5]) == 0:
                person_detections.append(det[:5])
            else:
                object_detections.append(det)
        # Update tracker with person detections only.
        tracker.update(frame, person_detections)

    # Clear map and button canvases.
    map_canvas.fill(0)
    button_canvas.fill(0)

    # -------------------------- SELECTION FOR TRACKED PERSONS --------------------------
    # Only applicable in People or Both modes.
    if detection_mode in (0, 2) and select_mode and click_position:
        for track in tracker.tracks:
            x1, y1, x2, y2 = track.bbox
            if x1 <= click_position[0] <= x2 and y1 <= click_position[1] <= y2:
                selected_track_id = track.track_id
                path_history = []  # Reset path history for new selection
                click_position = None
                break
        select_mode = False

    # -------------------------- DRAWING --------------------------
    # Draw detections/tracked objects on the video feed.
    if detection_mode == 0:
        # People only: draw tracked persons with bounding boxes and IDs.
        for track in tracker.tracks:
            x1, y1, x2, y2 = track.bbox
            track_id = track.track_id
            color = get_color(track_id, selected_track_id)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Map plotting for selected person
            cx = int((x1 + x2) // 2)
            cy = int((y1 + y2) // 2)
            norm_cx = int(cx / VIDEO_WIDTH * MAP_WIDTH)
            norm_cy = int(cy / VIDEO_HEIGHT * MAP_HEIGHT)
            if track_id == selected_track_id:
                path_history.append((norm_cx, norm_cy))
                triangle_pts = np.array([
                    (norm_cx, norm_cy - 10),
                    (norm_cx - 10, norm_cy + 10),
                    (norm_cx + 10, norm_cy + 10)
                ])
                cv2.drawContours(map_canvas, [triangle_pts], 0, color, -1)
            else:
                cv2.circle(map_canvas, (norm_cx, norm_cy), 10, color, -1)

    elif detection_mode == 1:
        # Object only: simply draw bounding boxes with class labels.
        for det in object_detections:
            x1, y1, x2, y2, score, class_id = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = class_names.get(class_id, f"ID {class_id}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else:
        # Both: draw tracked persons and simply draw objects.
        # Draw tracked persons.
        for track in tracker.tracks:
            x1, y1, x2, y2 = track.bbox
            track_id = track.track_id
            color = get_color(track_id, selected_track_id)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Map plotting for selected person.
            cx = int((x1 + x2) // 2)
            cy = int((y1 + y2) // 2)
            norm_cx = int(cx / VIDEO_WIDTH * MAP_WIDTH)
            norm_cy = int(cy / VIDEO_HEIGHT * MAP_HEIGHT)
            if track_id == selected_track_id:
                path_history.append((norm_cx, norm_cy))
                triangle_pts = np.array([
                    (norm_cx, norm_cy - 10),
                    (norm_cx - 10, norm_cy + 10),
                    (norm_cx + 10, norm_cy + 10)
                ])
                cv2.drawContours(map_canvas, [triangle_pts], 0, color, -1)
            else:
                cv2.circle(map_canvas, (norm_cx, norm_cy), 10, color, -1)

        # Draw non-person objects.
        for det in detections_all:
            if int(det[5]) != 0:  # non-person
                x1, y1, x2, y2, score, class_id = det
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                label = class_names.get(class_id, f"ID {class_id}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Draw path history for the selected tracked person.
    if detection_mode in (0, 2) and selected_track_id in [track.track_id for track in tracker.tracks]:
        for i in range(1, len(path_history)):
            cv2.line(map_canvas, path_history[i - 1], path_history[i], (0, 0, 255), 2)

    draw_grid(map_canvas, num_rows, num_cols, row_height, col_width)
    draw_buttons()

    # Combine video feed and button canvas to form the left panel.
    left_panel = np.vstack((frame, button_canvas))
    map_panel = cv2.resize(map_canvas, (MAP_WIDTH, left_panel.shape[0]))
    combined_interface = np.hstack((left_panel, map_panel))

    cv2.imshow("Main Interface", combined_interface)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
