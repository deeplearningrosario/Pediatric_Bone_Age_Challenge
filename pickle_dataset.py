#!/usr/bin/python3

from six.moves import cPickle
import cv2
import fnmatch
import numpy as np
import os
import pandas as pd
import sys
import math

# Turn saving renders feature on/off
SAVE_RENDERS = not False

# Create intermediate images in separate folders for debugger.
# mask, cut_hand, delete_object, render
SAVE_IMAGE_FOR_DEBUGGER = not False

# Extracting hands from images and using that new dataset.
# Simple dataset is correct, I am verifying the original.
EXTRACTING_HANDS = not False

# Turn rotate image on/off
ROTATE_IMAGE = False

# Usar el descriptor basado en gradiente
IMAGE_GRADIENTS = False


# Show the images
def writeImage(path, image):
    if SAVE_IMAGE_FOR_DEBUGGER:
        cv2.imwrite(
            os.path.join(__location__, "dataset_sample", path, img_file),
            image
        )


# XXX: Combertimos en gris y normalizamos el historama de colores
def histogramsEqualization(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


# Cut the hand of the image
# Look for the largest objects and create a mask, with that new mask
# is applied to the original and cut out.
def cutHand(image):
    image_copy = image.copy()

    img = cv2.medianBlur(image, 5)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

    # thresholdin: Otsu's Binarization method
    thresh = cv2.bitwise_not(th3, th2)
    thresh = cv2.GaussianBlur(thresh, (5, 5), 0)

    (_, contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # I guess the largest object is the hand or the only object in the image.
    largest_object_index = 0
    for i, cnt in enumerate(contours):
        if cv2.contourArea(contours[largest_object_index]) < cv2.contourArea(cnt):
            largest_object_index = i

    # create bounding rectangle around the contour (can skip below two lines)
    [x, y, w, h] = cv2.boundingRect(contours[largest_object_index])
    # White background below the largest object
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), -1)

    cv2.drawContours(
        image,  # image,
        contours,  # objects
        largest_object_index,  # índice de objeto (-1, todos)
        (255, 255, 255),  # color
        -1  # tamaño del borde (-1, pintar adentro)
    )

    # Trim that object of mask and image
    mask = image[y:y+h, x:x+w]
    image_cut = image_copy[y:y+h, x:x+w]

    # Apply mask
    image_cut = cv2.bitwise_and(image_cut, image_cut, mask=mask)

    writeImage("cut_hand", np.hstack([  # ================================
        image_cut,
    ]))  # show the images ===============================================
    return image_cut


# Create a mask for the hand.
# I guess the biggest objecUsar el descriptor basado en gradientet is the hand
def createMask(image, sensitivity=0.15):

    mask = cv2.inRange(
        image,
        np.array(int(sensitivity * 255)),  # lower color
        np.array(255)  # upper color
    )

    image_mask = cv2.bitwise_and(image, image, mask=mask)

    (_, contours, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # I guess the largest object is the hand or the only object in the image.
    largest_object_index = 0
    for i, cnt in enumerate(contours):
        if cv2.contourArea(contours[largest_object_index]) < cv2.contourArea(cnt):
            largest_object_index = i

    # create bounding rectangle around the contour (can skip below two lines)
    [x, y, w, h] = cv2.boundingRect(contours[largest_object_index])
    # White background below the largest object
    cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
    cv2.drawContours(
        mask,  # image,
        contours,  # objects
        largest_object_index,  # índice de objeto (-1, todos)
        (255, 255, 255),  # color
        -1  # tamaño del borde (-1, pintar adentro)
    )

    # Trim that object of mask and image
    mask = mask[y:y+h, x:x+w]
    image_cut = image[y:y+h, x:x+w]

    # Apply mask
    image_mask = cv2.bitwise_and(image_cut, image_cut, mask=mask)

    # NOTE ROTAR ACA ??

    # thresholdin: Otsu's Binarization method
    # _, mask = cv2.threshold(blur, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU )

    # Clean black spaces within the target
    # kernel = np.ones((25, 15), np.uint8)
    # image_mask = cv2.medianBlur(image_mask, 5)
    # image_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # image_mask = cv2.GaussianBlur(image_mask, (5, 5), 0)
    # kernel = np.ones((5, 5), np.uint8)
    # image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_OPEN, kernel)
    # kernel = np.ones((10, 10), np.uint8)
    # image_mask = cv2.erode(thresh, kernel, iterations=2)
    # kernel = np.ones((5, 5), np.uint8)
    # image_mask = cv2.dilate(image_mask, kernel, iterations=3)
    # image_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # image_mask = cv2.dilate(image_mask, kernel, iterations=1)

    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    # image_mask = cv2.dilate(image_mask, kernel, iterations=5)
    # image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_OPEN, kernel)
    # kernel = np.ones((5, 5), np.uint8)
    # image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_CLOSE, kernel)

    writeImage("mask", np.hstack([  # ====================================
        # image,
        mask,
        image_mask,
    ]))  # show the images ===============================================

    (_, contours, _) = cv2.findContours(image_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # He reported it because it can take a long time if the number is large
    if len(contours) > 5000:
        updateProgress(progress[0], progress[1], total_file,
                       img_file + " Img with " + str(len(contours)) + " contours")

    # I guess the largest object is the hand or the only object in the image.
    # From the contour list search the index of the largest object
    largest_object_index = 0
    for i, cnt in enumerate(contours):
        if cv2.contourArea(contours[largest_object_index]) < cv2.contourArea(cnt):
            largest_object_index = i

    # create bounding rectangle around the contour (can skip below two lines)
    [x, y, w, h] = cv2.boundingRect(contours[largest_object_index])
    # Black background below the largest object
    cv2.rectangle(image_mask, (x, y), (x+w, y+h), (0, 0, 0), -1)

    cv2.drawContours(
        image_mask,  # image,
        contours,  # objects
        largest_object_index,  # índice de objeto (-1, todos)
        (255, 255, 255),  # color
        -1  # tamaño del borde (-1, pintar adentro)
    )

    """
    # Clean black spaces within the target
    kernel = np.ones((20, 20), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Blur the image to avoid erasing borders
    mask = cv2.GaussianBlur(mask, (25, 25), 0)

    # Joining broken parts of an object.
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=2)

    # Detect the edges with Canny
    # img = cv2.Canny(image, 100, 100)  # 50,150  ; 100,500

    # _, thresh = cv2.threshold(image, 80, 255, cv2.THRESH_OTSU)

    # Delete other figures
    mask, contours = deleteObjects(mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Blur the image to avoid erasing borders
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    """

    return image


# Create and apply a mask to the image, then cut out the large objects and
# if the image is too white again, apply the procedure.
# In each iteration the intensity increases.
def extractingHands(image, sensitivity=0.20):
    # Create mask to highlight your hand
    mask = createMask(image, sensitivity)
    # Aplly mask
    img = cv2.bitwise_and(image, image, mask=mask)

    # avg_white = cv2.mean(img)[0]  # 0 to 255
    # print('\n value', avg_white, 'proce', avg_white *
    #      0.392, 'sen', sensitivity, '+', avg_white * 0.001)
    # if (avg_white * 0.392 > 60.0):
    #     img = extractingHands(image, sensitivity + avg_white * 0.001)

    return img


def rotateImage(imageToRotate):
    edges = cv2.Canny(imageToRotate, 50, 150, apertureSize=3)
    # Obtener una línea de la imágen
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(imageToRotate, (x1, y1), (x2, y2), (0, 0, 255), 2)
        angle = math.atan2(y1 - y2, x1 - x2)
        angleDegree = (angle*180)/math.pi

    if (angleDegree < 0):
        angleDegree = angleDegree + 360
    # print('\n', angleDegree)

    if (angleDegree >= 0 and angleDegree < 45):
        angleToSubtract = 0
    elif (angleDegree >= 45 and angleDegree < 135):
        angleToSubtract = 90
    elif (angleDegree >= 135 and angleDegree < 225):
        angleToSubtract = 180
    elif (angleDegree >= 225 and angleDegree < 315):
        angleToSubtract = 270
    else:
        angleToSubtract = 0
    # print(angleToSubtract)
    angleToRotate = angleDegree - angleToSubtract
    # print(angleToRotate)
    num_rows, num_cols = imageToRotate.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angleToRotate, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    return img_rotation


# Show a progress bar
def updateProgress(progress, tick='', total='', status='Loading...'):
    lineLength = 80
    barLength = 23
    if isinstance(progress, int):
        progress = float(progress)
    if progress < 0:
        progress = 0
        status = "Waiting...\r"
    if progress >= 1:
        progress = 1
        status = "Completed loading data\r\n"
    block = int(round(barLength * progress))
    line = str("\rImage: {0}/{1} [{2}] {3}% {4}").format(
        tick,
        total,
        str(("#" * block)) + str("." * (barLength - block)),
        round(progress * 100, 1),
        status
    )
    emptyBlock = lineLength - len(line)
    emptyBlock = " "*emptyBlock if emptyBlock > 0 else ""
    sys.stdout.write(line + emptyBlock)
    sys.stdout.flush()


# For this problem the validation and test data provided by the concerned authority did not have labels, so the training data was split into train, test and validation sets
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# __location__ = os.path.realpath(os.path.join(
#    os.getcwd(), os.path.dirname("C:/Pablo/Git/deeplearningforcomputervision/")))

train_dir = os.path.join(__location__, 'dataset_sample')

X_train = []
y_age = []
y_gender = []

df = pd.read_csv(os.path.join(train_dir, 'boneage-training-dataset.csv'))
a = df.values
m = a.shape[0]


# Create the directories to save the images
if SAVE_IMAGE_FOR_DEBUGGER:
    for folder in ['mask', 'cut_hand', 'delete_object', 'render']:
        if not os.path.exists(os.path.join(__location__, "dataset_sample", folder)):
            os.makedirs(os.path.join(__location__, "dataset_sample", folder))
if SAVE_RENDERS:
    if not os.path.exists(os.path.join(__location__, "dataset_sample", "render")):
        os.makedirs(os.path.join(__location__, "dataset_sample", "render"))

print('Loading data set...')
# file names on train_dir
files = os.listdir(train_dir)
# filter image files
files = [f for f in files if fnmatch.fnmatch(f, '*.png')]
total_file = len(files)

for i in range(total_file):
    img_file = files[i]

    # Update the progress bar
    progress = float(i / total_file), (i + 1)
    updateProgress(progress[0], progress[1], total_file, img_file)

    y_age.append(df.boneage[df.id == int(img_file[:-4])].tolist()[0])
    a = df.male[df.id == int(img_file[:-4])].tolist()[0]
    if a:
        y_gender.append(1)
    else:
        y_gender.append(0)

    # Read a image
    img_path = os.path.join(train_dir, img_file)
    img = cv2.imread(img_path)

    img = histogramsEqualization(img)

    if EXTRACTING_HANDS:
        # Trim the hand of the image
        img = cutHand(img)
        img = extractingHands(img)

    if ROTATE_IMAGE:
        # Rotate hands
        img = rotateImage(img)

    # TODO Image Gradients
    if IMAGE_GRADIENTS:
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        img = cv2.bitwise_or(sobelx, sobely)

        # sobelx8u = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)
        # sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        # abs_sobel64f = np.absolute(sobelx64f)
        # sobel_8u = np.uint8(abs_sobel64f)

    # ====================== show the images ================================
    if SAVE_IMAGE_FOR_DEBUGGER or SAVE_RENDERS:
        cv2.imwrite(
            os.path.join(__location__, "dataset_sample", "render", img_file),
            np.hstack([
                img
            ])
        )
    # =======================================================================

    # Resize the images
    img = cv2.resize(img, (224, 224))

    x = np.asarray(img, dtype=np.uint8)
    X_train.append(x)

updateProgress(1, total_file, total_file, img_file)

print('\nSaving data...')
# Save data
train_pkl = open('data.pkl', 'wb')
cPickle.dump(X_train, train_pkl, protocol=cPickle.HIGHEST_PROTOCOL)
train_pkl.close()

train_age_pkl = open('data_age.pkl', 'wb')
cPickle.dump(y_age, train_age_pkl, protocol=cPickle.HIGHEST_PROTOCOL)
train_age_pkl.close()

train_gender_pkl = open('data_gender.pkl', 'wb')
cPickle.dump(y_gender, train_gender_pkl, protocol=cPickle.HIGHEST_PROTOCOL)
train_gender_pkl.close()
print('\nCompleted saved data')
