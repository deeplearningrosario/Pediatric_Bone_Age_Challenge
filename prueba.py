"""
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

    edges = cv2.Canny(img_rotation,50,150,apertureSize = 3)
    minLineLength = 100
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(img_rotation,(x1,y1),(x2,y2),(0,255,0),2)

    num_rows, num_cols = imageToRotate.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 0, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
"""
import numpy as np
age_train = np.array([1, 2, 3, 4, 5, 6])
gdr_train = np.array([10, 20, 30, 40, 50, 60])

random_no = np.random.choice(age_train.shape[0], size=age_train.shape[0], replace=False)

print(gdr_train[0], age_train[0])
age_train = age_train[random_no]
gdr_train = gdr_train[random_no]
print(gdr_train[0], age_train[0])
