import cv2
import tensorflow as tf
import numpy as np
import dlib
from typing import List
import os

print(f"GPU setup complete!")

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("./predictor/shape_predictor_68_face_landmarks.dat")

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path:str) -> List[float]: 
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        faces = hog_face_detector(frame.numpy())
        lips = []
        for face in faces:
            face_landmarks = dlib_facelandmark(frame.numpy(), face)

            lips = []

            for n in range(48, 68):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                lips.append([x, y])
        if len(lips) != 0:
            lips = np.array(lips)
            centroid = np.mean(lips, axis = 0)
            start_point = (int(centroid[0] - 100 // 2), int(centroid[1] - 50 // 2))
            end_point = (int(centroid[0] + 100 // 2), int(centroid[1] + 50 // 2))
            frames.append(frame[start_point[1]:end_point[1],start_point[0]:end_point[0],:])
        else:
            print('No frames detected')
            frames.append(np.zeroes(shape = (50, 100, 1)))
            print(frames[-1])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

def load_alignments(path:str) -> List[str]: 
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
    #file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments