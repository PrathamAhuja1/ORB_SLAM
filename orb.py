import os
import cv2
import numpy as np
import orbslam3

# === Paths ===
VOCAB_PATH = "/home/pratham/Documents/ORB_SLAM3/Vocabulary/ORBvoc.txt"
VIDEO_PATH = "Video_2.mp4"
CONFIG_PATH = "config_example.yaml" 


DISPLAY_WIDTH = 1280    
WINDOW_HEIGHT = 720
PLAYBACK_SPEED = 1.5
DETECTION_METHOD = "background"
BG_SUB_HISTORY = 500
BG_SUB_VAR = 16
MIN_CONTOUR_AREA = 500
MAX_FEATURES, GOOD_MATCH_RATIO, MIN_OUTLIER_MATCHES = 500, 0.75, 10

def detect_motion_bg(frame, back_sub):
    fg = back_sub.apply(frame)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            return x, y, x+w, y+h
    return None

def detect_motion_feature(prev_gray, gray, orb_detector, bf):
    kp1, des1 = orb_detector.detectAndCompute(prev_gray, None)
    kp2, des2 = orb_detector.detectAndCompute(gray, None)
    if des1 is None or des2 is None:
        return None
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m,n in matches if m.distance < GOOD_MATCH_RATIO * n.distance]
    if len(good) < MIN_OUTLIER_MATCHES:
        return None
    pts = [kp2[m.trainIdx].pt for m in good]
    xs = [int(p[0]) for p in pts]
    ys = [int(p[1]) for p in pts]
    return (min(xs), min(ys), max(xs), max(ys)), gray

def main():
    SLAMClass = orbslam3.system
    slam = SLAMClass(VOCAB_PATH, CONFIG_PATH, orbslam3.Sensor.MONOCULAR)
    if hasattr(slam, "Initialize"):
        slam.Initialize()
    elif hasattr(slam, "initialize"):
        slam.initialize()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: cannot open {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = int((1.0 / fps) * 1000 / PLAYBACK_SPEED)

    window_name = "ORB-SLAM3 + Motion Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, WINDOW_HEIGHT)

    if DETECTION_METHOD == "background":
        back_sub = cv2.createBackgroundSubtractorMOG2(history=BG_SUB_HISTORY, varThreshold=BG_SUB_VAR)
    else:
        orb_detector = cv2.ORB_create(nfeatures=MAX_FEATURES)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        prev_gray = None

    trajectory = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        try:
            if hasattr(slam, "TrackMonocular"):
                cam_pose = slam.TrackMonocular(frame, timestamp)
            elif hasattr(slam, "track_monocular"):
                cam_pose = slam.track_monocular(frame, timestamp)
            else:
                cam_pose = None
        except Exception as e:
            print("SLAM error:", e)
            cam_pose = None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if DETECTION_METHOD == "background":
            box = detect_motion_bg(frame, back_sub)
        else:
            if prev_gray is None:
                prev_gray = gray
                box = None
            else:
                res = detect_motion_feature(prev_gray, gray, orb_detector, bf)
                box, prev_gray = res if res else (None, prev_gray)

        vis = frame.copy()
        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            trajectory.append((cx, cy))
        for i in range(1, len(trajectory)):
            cv2.line(vis, trajectory[i-1], trajectory[i], (255,0,0), 2)

        status = "SLAM OK" if cam_pose is not None else "SLAM LOST"
        color = (0,255,0) if cam_pose else (0,0,255)
        cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        h, w = vis.shape[:2]
        disp_w = DISPLAY_WIDTH // 2
        disp_h = int(h * (disp_w / w))
        combo = np.hstack([
            cv2.resize(frame, (disp_w, disp_h)),
            cv2.resize(vis, (disp_w, disp_h))
        ])
        cv2.imshow(window_name, combo)
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break

    if hasattr(slam, "Shutdown"):
        slam.Shutdown()
    elif hasattr(slam, "shutdown"):
        slam.shutdown()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
