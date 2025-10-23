import cv2

def analyze_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    altered_frames = []
    prev_frame = None

    for i in range(frame_count):
        ret, frame = cap.read()
        if ret:
            if prev_frame is not None:
                if not (frame == prev_frame).all():
                    altered_frames.append(i)
            prev_frame = frame.copy()

    cap.release()
    return altered_frames
