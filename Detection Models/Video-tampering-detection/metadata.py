import cv2

def extract_metadata(video_path):
    metadata = {}
    cap = cv2.VideoCapture(video_path)
    metadata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata['frame_width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    metadata['frame_height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    metadata['fps'] = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return metadata
