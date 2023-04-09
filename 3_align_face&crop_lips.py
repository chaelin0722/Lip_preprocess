import numpy as np
import cv2
import os
import math
from tqdm import tqdm

from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor
from PIL import Image
import shutil
from skimage import transform as trans
from facial_analysis import FacialImageProcessing
imgProcessing=FacialImageProcessing(False)

DIR = "/faces/"
OUTPUT_DIR = "align/"
LIP_OUT_DIR = "lips/"
def get_iou(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

#print(get_iou([10,10,20,20],[15,15,25,25]))

def preprocess(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = [224,224]
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==224:
        src[:,0] += 8.0
    src*=2
    if landmark is not None:
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        #dst=dst[:3]
        #src=src[:3]
        #print(dst.shape,src.shape,dst,src)
        tform.estimate(dst, src)
        M = tform.params[0:2,:]
        #M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)
        #print(M)

    if M is None:
        if bbox is None: #use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
              det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin//2, 0)
        bb[1] = np.maximum(det[1]-margin//2, 0)
        bb[2] = np.minimum(det[2]+margin//2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin//2, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        if len(image_size)>0:
              ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else: #do align using landmark
        assert len(image_size)==2
        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
        return warped


def crop_lip(image, landmarks, output_dir, folder_dir):
    image_dir =output_dir
    image = Image.open(image)

    x1, _ = landmarks[49]
    x2, _ = landmarks[55]
    _, y1 = landmarks[51]
    _, y2 = landmarks[58]

    width, height = image.size
    width_add = int((88-(x2-x1))/2)
    height_add = int((88-(y2-y1))/2)

    if x1 - width_add <= 0:
        x1 = 1
    else:
        x1 = x1 - width_add

    if x2 + width_add > width:
        x2 = width
    else:
        x2 = x2 + width_add

    if y1 - height_add <= 0:
        y1 = 1
    else:
        y1 = y1 - height_add

    if y2 + height_add > height:
        y2 = height
    else:
        y2 = y2 + height_add

    try:
        cropImage = image.crop((x1, y1 , x2, y2 ))
        cropImage.save(image_dir)
    except:

        files = os.listdir(folder_dir)
        files.sort()

        shutil.copyfile(os.path.join(folder_dir + "/" + files[-1]),
                        os.path.join(folder_dir + "/" + image))

    #cropImage = cropImage.resize((32, 32))
    #cropImage = np.array(cropImage)
    #cropImage = cropImage[np.newaxis]



def save_aligned_faces(source_path,save_path, lip_save_path):

    for sub in ['train', 'val', 'test']:
        if not os.path.exists(os.path.join(save_path, sub)):
            os.mkdir(os.path.join(save_path, sub))

        if not os.path.exists(os.path.join(lip_save_path, sub)):
            os.mkdir(os.path.join(lip_save_path, sub))

        for folder1 in os.listdir(os.path.join(source_path, sub)):
            if not os.path.exists(os.path.join(save_path + sub + "/" + folder1)):
                os.mkdir(os.path.join(save_path + sub + "/" + folder1))

            if not os.path.exists(os.path.join(lip_save_path + sub + "/" + folder1)):
                os.mkdir(os.path.join(lip_save_path + sub + "/" + folder1))



            source_folder_path = os.path.join(source_path + sub + "/" + folder1)
            output_folder_path = os.path.join(save_path + sub + "/" + folder1)
            lip_output_folder_path = os.path.join(lip_save_path + sub + "/" + folder1)

            for folder2 in os.listdir(source_folder_path):
                if not os.path.exists(os.path.join(output_folder_path, folder2)):
                    os.mkdir(os.path.join(output_folder_path, folder2))

                if not os.path.exists(os.path.join(lip_output_folder_path, folder2)):
                    os.mkdir(os.path.join(lip_output_folder_path, folder2))

                source_folder_path2 = os.path.join(source_folder_path + "/" + folder2)
                output_folder_path2 = os.path.join(output_folder_path + "/" + folder2)
                lip_output_folder_path2 = os.path.join(lip_output_folder_path + "/" + folder2)
                images = os.listdir(source_folder_path2)
                images.sort()
                for image in images:
                    if len(images) == len(os.listdir(output_folder_path2)) and \
                            len(images) == len(os.listdir(lip_output_folder_path2)):

                        ddd = os.path.join(output_folder_path + "/" + folder2)
                        print("same, skip", ddd)
                        continue

                    filename = os.path.join(source_folder_path2 + "/" + image)
                    frame = cv2.imread(filename)
                    #frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    bounding_boxes, points = imgProcessing.detect_faces(frame)
                    points = points.T

                    best_ind=None
                    prev_b = None
                    counter=0

                    try:
                        best_ind=0
                        b=[int(bi) for bi in bounding_boxes[best_ind]]
                        counter=0
                        if True:
                            p = None
                            if best_ind is not None:
                                p = points[best_ind]
                                if True:  # not USE_RETINA_FACE:
                                    p = p.reshape((2, 5)).T
                            face_img = preprocess(frame, b, p)  # None, #p)
                        else:
                            x1, y1, x2, y2 = b[0:4]
                            face_img = frame[y1:y2, x1:x2, :]

                        cv2.imwrite(os.path.join(output_folder_path2 + "/" + image), face_img)
                        ## for crop lips
                        detected_faces = face_detector(face_img, rgb=True)
                        landmarks, scores = landmark_detector(face_img, detected_faces, rgb=False)
                        lip_image = lip_output_folder_path2 + "/" + image
                        source_image = os.path.join(output_folder_path2 + "/" + image)
                        crop_lip(source_image, landmarks[0], lip_image, output_folder_path2)

                    except:
                        print("no bbox, copy original face image", filename)

                        frame = cv2.resize(frame, (224,224))
                        cv2.imwrite(os.path.join(output_folder_path2 + "/" + image), frame)


                        detected_faces = face_detector(frame, rgb=True)
                        landmarks, scores = landmark_detector(frame, detected_faces, rgb=False)
                        if len(landmarks) == 0:
                            try:
                                lip_files = os.listdir(lip_output_folder_path2)
                                lip_files.sort()
                                shutil.copyfile(os.path.join(lip_output_folder_path2 + "/" + lip_files[-1]),
                                                os.path.join(lip_output_folder_path2 + "/" + image))

                            except:
                                print("no pre-lips to copy")
                                continue

                        else:
                            lip_image = os.path.join(lip_output_folder_path2 + "/" + image)
                            source_image = os.path.join(output_folder_path2 + "/" + image)
                            crop_lip(source_image, landmarks[0], lip_image, output_folder_path2)


face_detector = RetinaFacePredictor(
    threshold=0.8, device='cuda:2',
    model=RetinaFacePredictor.get_model('resnet50'))

# Create a facial landmark detector
landmark_detector = FANPredictor(
    device='cuda:2', model=FANPredictor.get_model('2dfan2_alt'))

#aligned
save_aligned_faces(DIR, OUTPUT_DIR, LIP_OUT_DIR)
