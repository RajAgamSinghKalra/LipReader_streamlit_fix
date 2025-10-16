'''import tensorflow as tf
from typing import List
import cv2
import os
import dlib
import numpy as np

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path: str):
    cap = cv2.VideoCapture(path)
    frames_list = []
    
    MOUTH_POINTS = list(range(48, 68))
    mouth_coords = None
    
    # First pass: Find mouth position in any frame
    while cap.isOpened() and mouth_coords is None:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)  # Use upsampling for accuracy in first frame
        
        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)
            
            # Get mouth coordinates from this frame
            mouth_coords = np.array([[landmarks.part(i).x, landmarks.part(i).y] 
                                   for i in MOUTH_POINTS])
            
            # Calculate mouth bounding box
            margin = 15
            x_min = np.min(mouth_coords[:, 0]) - margin
            x_max = np.max(mouth_coords[:, 0]) + margin
            y_min = np.min(mouth_coords[:, 1]) - margin
            y_max = np.max(mouth_coords[:, 1]) + margin
            
            # Ensure coordinates within frame bounds
            h, w = gray.shape
            x_min = max(0, x_min)
            x_max = min(w, x_max)
            y_min = max(0, y_min)
            y_max = min(h, y_max)
            
            # Store the mouth region coordinates
            mouth_region = (x_min, y_min, x_max, y_max)
            break
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Second pass: Apply the same mouth crop to all frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if mouth_coords is not None:
            # Use the detected mouth position
            x_min, y_min, x_max, y_max = mouth_region
        else:
            # Fallback: Use manual crop (center of frame)
            h, w = gray.shape
            x_min, x_max = w//2 - 50, w//2 + 50
            y_min, y_max = h//2 - 25, h//2 + 25
        
        # Ensure coordinates are valid
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        
        # Crop and resize
        cropped_mouth = gray[y_min:y_max, x_min:x_max]
        cropped_mouth = cv2.resize(cropped_mouth, (120, 60))
        cropped_mouth = np.expand_dims(cropped_mouth, axis=-1)
        
        frames_list.append(cropped_mouth)
    
    cap.release()
    
    # Ensure we have exactly 75 frames
    if len(frames_list) == 0:
        return tf.zeros([75, 60, 120, 1], dtype=tf.float32)
    
    # Pad or truncate to 75 frames
    if len(frames_list) < 75:
        last_frame = frames_list[-1] if frames_list else np.zeros((50, 100, 1))
        while len(frames_list) < 75:
            frames_list.append(last_frame.copy())
    else:
        frames_list = frames_list[:75]
    
    # Normalize
    frames_tensor = tf.constant(frames_list, dtype=tf.float32)
    mean = tf.math.reduce_mean(frames_tensor)
    std = tf.math.reduce_std(frames_tensor)
    epsilon = 1e-6
    normalized_frames = (frames_tensor - mean) / (std + epsilon)
    
    return normalized_frames

def load_alignments(path:str) -> List[str]: 
    #print(path)
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    # File name splitting for windows
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('..','data','all',f'{file_name}.mpg')
    alignment_path = os.path.join('..','data','align',f'{file_name}.align')
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)
    
    return frames, alignments'''


import tensorflow as tf
from typing import List
import cv2
import os
import dlib
import numpy as np

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path: str):
    cap = cv2.VideoCapture(path)
    frames_list = []
    
    MOUTH_POINTS = list(range(48, 68))
    
    # Get total frames and FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate how many 75-frame segments we need
    num_segments = max(1, (total_frames + 74) // 75)  # Ceiling division
    
    for segment in range(num_segments):
        mouth_coords = None
        mouth_region = None
        
        # Find mouth position in first frame of each segment
        for offset in range(min(10, 75)):  # Try first 10 frames of segment
            frame_pos = segment * 75 + offset
            if frame_pos >= total_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 1)
            
            if len(faces) > 0:
                face = faces[0]
                landmarks = predictor(gray, face)
                
                # Get mouth coordinates
                mouth_coords = np.array([[landmarks.part(i).x, landmarks.part(i).y] 
                                       for i in MOUTH_POINTS])
                
                # Calculate mouth bounding box
                margin = 15
                x_min = np.min(mouth_coords[:, 0]) - margin
                x_max = np.max(mouth_coords[:, 0]) + margin
                y_min = np.min(mouth_coords[:, 1]) - margin
                y_max = np.max(mouth_coords[:, 1]) + margin
                
                # Ensure coordinates within frame bounds
                h, w = gray.shape
                x_min = max(0, x_min)
                x_max = min(w, x_max)
                y_min = max(0, y_min)
                y_max = min(h, y_max)
                
                mouth_region = (x_min, y_min, x_max, y_max)
                break
        
        # Process 75 frames for this segment
        segment_frames = []
        for i in range(75):
            frame_pos = segment * 75 + i
            if frame_pos >= total_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if mouth_region is not None:
                # Use the detected mouth position
                x_min, y_min, x_max, y_max = mouth_region
            else:
                # Fallback: Use manual crop (center of frame)
                h, w = gray.shape
                x_min, x_max = w//2 - 60, w//2 + 60
                y_min, y_max = h//2 - 30, h//2 + 30
            
            # Ensure coordinates are valid
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            
            # Crop and resize
            cropped_mouth = gray[y_min:y_max, x_min:x_max]
            cropped_mouth = cv2.resize(cropped_mouth, (120, 60))
            cropped_mouth = np.expand_dims(cropped_mouth, axis=-1)
            
            segment_frames.append(cropped_mouth)
        
        # Pad segment to exactly 75 frames if needed
        if len(segment_frames) < 75 and len(segment_frames) > 0:
            last_frame = segment_frames[-1]
            while len(segment_frames) < 75:
                segment_frames.append(last_frame.copy())
        
        frames_list.extend(segment_frames[:75])  # Take exactly 75 frames per segment
    
    cap.release()
    
    # If no frames processed, return dummy data
    if len(frames_list) == 0:
        return tf.zeros([75, 60, 120, 1], dtype=tf.float32)
    
    # Convert to tensor and normalize
    frames_tensor = tf.constant(frames_list, dtype=tf.float32)
    mean = tf.math.reduce_mean(frames_tensor)
    std = tf.math.reduce_std(frames_tensor)
    epsilon = 1e-6
    normalized_frames = (frames_tensor - mean) / (std + epsilon)
    
    return normalized_frames

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    # For uploaded files, use the path directly
    if path.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        video_path = path
    else:
        # For legacy support with your existing file structure
        file_name = path.split('\\')[-1].split('.')[0]
        video_path = os.path.join('..','data','all',f'{file_name}.mpg')
    
    frames = load_video(video_path)
    
    # Return only frames (no alignments)
    return frames