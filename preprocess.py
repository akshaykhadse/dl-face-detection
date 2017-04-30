from PIL import Image
import os
import numpy as np
from PIL import ImageFile
import cv2

'''
Calculates the intersection over union of two regions in the image.
'''

def inter_over_uni(region1, region2):
    m1 = ((region1[0], region1[1]),  (region1[2], region1[1]), (region1[2],region1[3]),(region1[0], region1[3]))
    m2 = ((region2[0], region2[1]), (region2[2], region2[1]), (region2[2],region2[3]), (region2[0], region2[3]))
    result = []
    intersection = 0.0
    area = 2*abs((m1[1][0] - m1[0][0]) * (m1[3][1] - m1[0][1]))
    for p in m1:
        if(p[0] >= m2[0][0] and p[0] <= m2[1][0] and p[1] >= m2[0][1] and p[1] <= m2[3][1]):
            result.append(p)
    for p in m2:
        if(p[0] >= m1[0][0] and p[0] <= m1[1][0] and p[1] >= m1[0][1] and p[1] <= m1[3][1]):
            result.append(p)
    if(len(result) == 2):
        intersection = abs((result[1][0] - result[0][0]) * (result[1][1] - result[0][1]))
    elif(len(result) == 4):
        if(result[0][0] != result[1][0] and result[0][1] != result[1][1]):
            intersection = abs((result[1][0] - result[0][0]) * (result[1][1] - result[0][1]))
        elif(result[0][0] != result[2][0] and result[0][1] != result[2][1]):
            intersection = abs((result[2][0] - result[0][0]) * (result[2][1] - result[0][1]))
        elif(result[0][0] != result[3][0] and result[0][1] != result[3][1]):
            intersection = abs((result[3][0] - result[0][0]) * (result[3][1] - result[0][1]))
    return intersection/float(area - intersection)

'''
Function creates windows from each image. Checks IoU of the window with the faces.
Classifes images as postive or negative examples depending on the Iou value
'''
def crop(l):
    global negative_count
    global face
    l = l.split(',')
    index, image_name = l[0], l[1]
    num_faces = len(face[image_name])
    x, y, w, h = face[image_name][0][0], face[image_name][0][1], face[image_name][0][2], face[image_name][0][3]
    im = cv2.imread(image_name) #load in grayscale
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #load the colour version
    step_x, step_y = int(w/3), int(h/3)
    for x_dir in range(0, im.shape[1]-w+1, step_x):
        for y_dir in range(0, im.shape[0]-h+1, step_y):
            if num_faces == 1:
                iou = 0
                iou = inter_over_uni([x_dir, y_dir, x_dir+w, y_dir+h],[x, y, x+w, y+h])
            elif num_faces > 1:
                iou = 0
                for n in range(num_faces):
                    new_x, new_y = face[image_name][n][0], face[image_name][n][1]
                    new_w, new_h = face[image_name][n][2], face[image_name][n][3]
                    iou1 = inter_over_uni((x_dir, y_dir, x_dir+w, y_dir+h),(new_x, new_y, new_x+new_w, new_y+new_h))
                    if iou1 > iou:
                        iou = iou1
            if iou < 0.2:
                os.chdir('negative_bw')
                image_cropped = np.copy(im[y_dir:y_dir+h, x_dir:x_dir+w])
                to_size = 32
                image_rescaled = cv2.resize(image_cropped, (to_size,to_size), interpolation = cv2.INTER_AREA)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
                image_normalised = clahe.apply(image_rescaled)

                negative_count += 1
                cv2.imwrite(str(negative_count)+".png", image_normalised)

                os.chdir('..')

    print(index)


def compact(l):
    done = []
    global face
    for ind in range(len(l)):
        li = l[ind][:-1].split(',')
        name = li[1]
        x, w = int(li[2]), int(li[4])
        y, h = int(li[3]), int(li[5])
        if name not in done:
            done.append(name)
            face[name] = [[x, y, w, h]]
        else:
            face[name].append([x, y, w, h])

cropped = []
face = {}

negative_count = 0
os.chdir('../Datasets/')
if os.path.exists('negative_bw') == False:
    os.mkdir('negative_bw')
output = open('output/output.txt', 'r')
lines = output.readlines()
compact(lines)
for ind in range(len(lines)):
    if negative_count > 120000:
        print("Done")
        break
    if lines[ind][:-1].split(',')[1] in cropped:
        continue
    cropped.append(lines[ind][:-1].split(',')[1])
    crop(lines[ind][:-1])
