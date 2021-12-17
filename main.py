import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pydicom as dicom
import cv2
from PIL import Image

def RLenc(img,order='F',format=True):
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = [] 
    r = 0  
    pos = 1   
    for c in bytes:
        if ( c == 0 ):
            if r != 0:
                runs.append((pos, r))
                pos+=r
                r=0
            pos+=1
        else:
            r+=1

    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''
    
        for rr in runs:
            z+='{} {} '.format(rr[0],rr[1])
        return z[:-1]
    else:
        return runs

print("введите путь к файлу:")
image_path = input()

print("введите количество точек:")
point_n = input()

points_lst = []
for i in range(int(point_n)):
    print("введите x:")
    x = int(input())
    print("введите y:")
    y = int(input())

    xy = [x,y]
    points_lst.append3(xy)
    

points = np.array([points_lst])

image_path = 'data/sample.dcm'
data_set = dicom.dcmread(image_path)
dcm_sample = dcm_sample = np.array(data_set.pixel_array*128)


# maska
height = dcm_sample.shape[0]
width = dcm_sample.shape[1]

mask = np.zeros((height, width), dtype=np.uint8)

cv2.fillPoly(mask, points, (255))

res = cv2.bitwise_and(dcm_sample, dcm_sample, mask = mask)

rect = cv2.boundingRect(points) 
cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

cv2.imshow('sample', dcm_sample)
cv2.imshow('cropped', cropped)
cv2.imwrite('D:/image-2.png', cropped)
cv2.waitKey(0)

# plot output
img = data_set.pixel_array
plt.imshow(img, cmap='gray')

ax = plt.gca()

ax.add_patch(
     patches.Polygon(
        xy = points_lst,
        edgecolor = 'blue',
        facecolor = 'red',
        fill = False
)   )

plt.show()

print("Сохранить rle файл: (Y/N)")
ch = input()
if (ch == "Y" or ch == "y"):
    mask_rle = RLenc(cropped)
    rle_file = open("RleResult.txt", "w+")
    rle_file.write(mask_rle)
    rle_file.close()
    print(mask_rle[:100])

print("Сохранить png файл (D:/Result.png): (Y/N)")
ch = input()
if (ch == "Y" or ch == "y"):
    cv2.imwrite('D:/Result.png', cropped)
    cv2.waitKey(0)