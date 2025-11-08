#base_cv.py
import sys
import time
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

YELLOW = (0, 255, 255)
CIRCLE_RADIUS = 4
LINE_THICKNESS = 4
VISIBILITY_THRESH = 0.3

#copied this function
def draw_pose_on_frame(frame, results):
    h, w = frame.shape[:2]
    lm = results.pose_landmarks.landmark
    for a, b in POSE_CONNECTIONS:
        p1 = lm[a]
        p2 = lm[b]
        if(p1.visibility < VISIBILITY_THRESH) or (p2.visibility < VISIBILITY_THRESH):
            continue
        #pretty sure this just does lines, for each set of (a,b)
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), YELLOW, LINE_THICKNESS, lineType=cv2.LINE_AA)
    for lm_pt in lm:
        if lm_pt.visibility < VISIBILITY_THRESH:
            continue
        cx, cy = int(lm_pt.x * w), int(lm_pt.y * h)
        cv2.circle(frame, (cx, cy), CIRCLE_RADIUS, YELLOW, -1, lineType=cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("no camera") 
        sys.exit(1)

    #pose_setup
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        try:
            prev = time.time() #do fps calc
            while True:
                ret, frame = cap.read() #ret = did it work, frame = image
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                if results.pose_landmarks:
                    #draw the detected poses on the frame 
                    draw_pose_on_frame(frame, results)
                fps = 1.0 / (time.time() - prev) if time.time() != prev else 0.0
                prev = time.time()
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.imshow("Pose - yellow lines", frame)
                #q to quit
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
