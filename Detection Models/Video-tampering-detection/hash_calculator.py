import hashlib

def calculate_hash(video_path):
    hash_md5 = hashlib.md5()
    with open(video_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
