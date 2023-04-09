
import os
import shutil
import cv2
import numpy as np

from PIL import Image
from typing import Optional, Sequence, Tuple

from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor
from ibug.face_alignment.utils import plot_landmarks

from facial_analysis import FacialImageProcessing
imgProcessing=FacialImageProcessing(False)

INPUT_SIZE = (224, 224)
DATA_DIR = "frames/"
OUTPUT_DIR = "faces/"

def get_face_bbox(image, landmarks, output_dir):
    image_dir = image
    image = Image.open(image)

    x1, _ = landmarks[1]
    x2, _ = landmarks[16]
    _, y1 = landmarks[19]
    _, y2 = landmarks[8]

    try:
        cropImage = image.crop((x1 - 10, y1 - 10, x2 + 10, y2 + 10))
        #cropImage = cropImage.resize((32, 32))
        #cropImage = np.array(cropImage)
        #cropImage = cropImage[np.newaxis]
        imagename = image_dir.split("/")[-1]
        cropImage.save(os.path.join(output_dir +"/"+ imagename))

    except:

        files = os.listdir(output_dir)
        files.sort()

        shutil.copyfile(os.path.join(output_dir + "/" + files[-1]),
                        os.path.join(output_dir + "/" + image_dir.split("/")[-1]))

    #return [(x1 - 10, y1 - 10, x2 + 10, y2 + 10)]

def save_faces(source_path, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for sub in ['train', 'val', 'test']:
        if not os.path.exists(os.path.join(save_path, sub)):
            os.mkdir(os.path.join(save_path, sub))

        for folder1 in os.listdir(os.path.join(source_path, sub)):
            if not os.path.exists(os.path.join(save_path + sub+ "/"+ folder1)):
                os.mkdir(os.path.join(save_path+ sub+ "/"+ folder1))

            source_folder_path = os.path.join(source_path+ sub+ "/"+ folder1)
            output_folder_path = os.path.join(save_path+ sub+ "/"+ folder1)

            for folder2 in os.listdir(source_folder_path):
                if not os.path.exists(os.path.join(output_folder_path, folder2)):
                    os.mkdir(os.path.join(output_folder_path, folder2))

                source_folder_path2 = os.path.join(source_folder_path+ "/"+ folder2)
                output_folder_path2 = os.path.join(output_folder_path+ "/"+ folder2)

                if len(os.listdir(source_folder_path2)) == len(os.listdir(output_folder_path2)):
                    print("already existed :", output_folder_path2)
                    continue

                for image in os.listdir(source_folder_path2):

                    filename = os.path.join(source_folder_path2+ "/"+ image)
                    frame_bgr = cv2.imread(filename)
                    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                    # Detect faces from the image
                    detected_faces = face_detector(frame, rgb=True)

                    landmarks, scores = landmark_detector(frame, detected_faces, rgb=False)
                    try :
                        get_face_bbox(os.path.join(source_folder_path2+ "/"+ image), landmarks[0], output_folder_path2)
                    except:
                        img = os.path.join(source_folder_path2+ "/"+ image)
                        print("no face founded, copy pre image : ", img)

                        files = os.listdir(source_folder_path2)
                        files.sort()
                        if len(files) == 0 :
                            print("no pre image... skip this one : ", img)
                            continue

                        shutil.copyfile(os.path.join(source_folder_path2 + "/" + files[-1]),
                                        os.path.join(output_folder_path2 + "/" + image))

    else:
        print(save_path)

face_detector = RetinaFacePredictor(
    threshold=0.8, device='cuda:2',
    model=RetinaFacePredictor.get_model('resnet50'))

# Create a facial landmark detector
landmark_detector = FANPredictor(
    device='cuda:2', model=FANPredictor.get_model('2dfan2_alt'))

save_faces(DATA_DIR, OUTPUT_DIR)
