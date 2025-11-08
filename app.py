import os
import cv2
import mediapipe as mp
import json
import numpy as np

mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
CIRCLE_RADIUS = 4
LINE_THICKNESS = 4
VISIBILITY_THRESH = 0.3

POSE_OUTPUT_DIR = "pose_outputs"
POSE_CACHE_DIR = "pose_cache"

os.makedirs(POSE_CACHE_DIR, exist_ok=True)

def draw_pose(frame, landmarks, color):
    h, w = frame.shape[:2]
    for a, b in POSE_CONNECTIONS:
        p1 = landmarks[a]
        p2 = landmarks[b]
        if p1.visibility < VISIBILITY_THRESH or p2.visibility < VISIBILITY_THRESH:
            continue
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), color, LINE_THICKNESS, lineType=cv2.LINE_AA)
    for lm in landmarks:
        if lm.visibility < VISIBILITY_THRESH:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), CIRCLE_RADIUS, color, -1, lineType=cv2.LINE_AA)
    return frame

def load_and_align_pose(landmarks, user_nose, size):
    if not landmarks:
        return None

    class FakeLandmark:
        def __init__(self, x, y, z, v):
            self.x, self.y, self.z, self.visibility = x, y, z, v

    lm_list = [FakeLandmark(*pt) for pt in landmarks]
    h, w = size[1], size[0]

    if user_nose is None:
        return draw_pose(np.zeros((h, w, 3), dtype=np.uint8), lm_list, GREEN)

    athlete_nose = lm_list[0]
    dx = user_nose[0] - athlete_nose.x
    dy = user_nose[1] - athlete_nose.y

    for lm in lm_list:
        lm.x += dx
        lm.y += dy

    blank = np.zeros((h, w, 3), dtype=np.uint8)
    return draw_pose(blank, lm_list, GREEN)

def all_points_close(user_pts, athlete_pts, thresh=0.1):
    if user_pts is None or athlete_pts is None:
        return False
    total = 0
    match = 0
    for up, ap in zip(user_pts, athlete_pts):
        if up.visibility < VISIBILITY_THRESH or ap[3] < VISIBILITY_THRESH:
            continue
        dist = ((up.x - ap[0]) ** 2 + (up.y - ap[1]) ** 2) ** 0.5
        total += 1
        if dist < thresh:
            match += 1
    return total > 0 and match == total

def overlay_pose_video_with_user():
    exercise_folder = os.path.join("archive", "raw_data", "raw_data", "data-crawl")
    exercise_dirs = sorted(os.listdir(exercise_folder))
    if not exercise_dirs:
        print("No exercise folders found.")
        return

    for first_exercise in exercise_dirs:
        exercise_name = first_exercise.replace("_", " ").title()
        video_dir = os.path.join(exercise_folder, first_exercise)
        videos = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
        if not videos:
            continue

        sample_path = os.path.join(video_dir, videos[0])
        print(f"Using athlete video: {sample_path}")

        cache_name = os.path.splitext(os.path.basename(sample_path))[0] + ".npy"
        cache_path = os.path.join(POSE_CACHE_DIR, cache_name)
        if not os.path.exists(cache_path):
            print("Pose cache not found. Processing and saving...")
            try:
                process_and_cache_pose(sample_path, cache_path)
            except Exception as e:
                print(f"Skipping {sample_path} due to error during processing: {e}")
                try:
                    os.remove(sample_path)
                except: pass
                continue

        try:
            athlete_landmarks = np.load(cache_path, allow_pickle=True)
        except Exception as e:
            print(f"Skipping {sample_path} due to load error: {e}")
            try:
                os.remove(sample_path)
                os.remove(cache_path)
            except: pass
            continue

        cap_user = cv2.VideoCapture(0)
        user_pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        frame_idx = 0
        last_user_nose = None

        while cap_user.isOpened():
            ret2, user_frame = cap_user.read()
            if not ret2:
                break

            user_rgb = cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB)
            user_result = user_pose.process(user_rgb)
            user_landmarks = user_result.pose_landmarks.landmark if user_result.pose_landmarks else None

            h, w = user_frame.shape[:2]
            overlay_bar_h = 40
            display_frame = user_frame[overlay_bar_h:, :, :]
            resized = cv2.resize(display_frame, (w, h - overlay_bar_h))

            overlay_frame = resized.copy()
            if user_landmarks:
                overlay_frame = draw_pose(overlay_frame, user_landmarks, RED)

            aligned = load_and_align_pose(
                athlete_landmarks[frame_idx] if frame_idx < len(athlete_landmarks) else None,
                last_user_nose,
                (w, h - overlay_bar_h)
            )
            if aligned is not None:
                overlay_frame = cv2.addWeighted(overlay_frame, 0.6, aligned, 0.4, 0)

            if user_landmarks:
                last_user_nose = (user_landmarks[0].x, user_landmarks[0].y)

            user_close = all_points_close(user_landmarks, athlete_landmarks[frame_idx] if frame_idx < len(athlete_landmarks) else None)
            status = "Yay!" if user_close else "Keep Going"

            header = np.full((overlay_bar_h, w, 3), 255, dtype=np.uint8)
            cv2.putText(header, f"{exercise_name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(header, status, (w - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)

            full_frame = np.vstack((header, overlay_frame))
            cv2.imshow("User (Red) + Athlete Pose (Green)", full_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key in (ord("p"), ord("P")):
                print(f"Deleting: {sample_path}")
                try:
                    os.remove(sample_path)
                    os.remove(cache_path)
                    print("Deleted sample video and pose cache.")
                except Exception as e:
                    print(f"Error deleting files: {e}")
                break

            frame_idx = min(frame_idx + 1, len(athlete_landmarks) - 1)

        cap_user.release()
        cv2.destroyAllWindows()
        break

def process_and_cache_pose(video_path, cache_path):
    cap = cv2.VideoCapture(video_path)
    result = []
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = pose.process(rgb)
            if output.pose_landmarks:
                result.append([(lm.x, lm.y, lm.z, lm.visibility) for lm in output.pose_landmarks.landmark])
            else:
                result.append(None)
    cap.release()
    if not result or all(x is None for x in result):
        raise ValueError("No pose landmarks found in video.")
    np.save(cache_path, np.array(result, dtype=object))

print("Use p to delete an exercise from the dataset, and l to skip to the next exercise.")
print("The pose of the person doing the workout will follow you, so change your camera angle")
print("so it lines up nicely. You will get a Yay! if you are close enough else Keep going.")
print("name of exercise is in top left. you can press c to change to the next exercise")
print("if you're downloading this from github, you need to get your own Kaggle api from kaggle.com.")
print("then put this .py and workout_index.json in the same folder, then run this .py")
if __name__ == "__main__":
    overlay_pose_video_with_user()
