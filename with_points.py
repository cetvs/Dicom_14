import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pydicom as dicom
import cv2
from pathlib import Path
from tkinter import filedialog

# кодировка в rle формат
def rle_code(img):
    height = img.shape[0]
    width = img.shape[1]
    
    res = []
    count = 1

    for h in range(height):
        res_w = []
        for w in range(1, width):
            if (img[h,w - 1] != img[h,w]):
                res_w.append(count)
                res_w.append(img[h,w - 1])
                count = 1
            else:
                count += 1    
                
            if(w == width-1):
                res_w.append(count)
                res_w.append(img[h,w])
        res.append(res_w)    
    return res    

# сохранение в rle.txt
def rle_to_txt(rle, path):
    with open(path, 'w') as f:
        for lst in rle:
            last = len(lst) - 1

            for i in range(last):
                f.write(str(lst[i]) + " ")  

            f.write(str(lst[last]))  
            f.write( "\n" )     

def write_points():
    # ввод количества точек
    print("введите количество точек:")
    point_n = 3
    point_n = input()

    # ввод координат многоугольника
    points_lst = []
    for i in range(int(point_n)):
        print("введите x:")
        x = int(input())
        print("введите y:")
        y = int(input())
        xy = [x,y]
        points_lst.append(xy)
    return points_lst    

def image_cropped(dcm_sample, points):
    height = dcm_sample.shape[0]
    width = dcm_sample.shape[1]
    print("Размеры изображения: " + str(height) +" x "+ str(width))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, points, (255))
    res = cv2.bitwise_and(dcm_sample, dcm_sample, mask = mask)

    rect = cv2.boundingRect(points) 
    return res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
  

# ввод пути к файлу
image_path = filedialog.askopenfilename(initialdir="/", title="Select dicom file",
                    filetypes=(("DICOM files", "*.dcm"),("all files", "*.*")))

data_set = dicom.dcmread(image_path)

# image_path = 'data/sample.dcm'
dcm_sample = np.array(data_set.pixel_array*256)
#  points_lst = write_points()
points_lst = [[200, 200], [400, 350], [300,500]]

points = np.array([points_lst])
cropped = image_cropped(dcm_sample, points)


# plot output
img = dicom.pixel_data_handlers.util.apply_voi_lut(data_set.pixel_array, data_set, index=0)
plt.imshow(img, cmap='gray')

ax = plt.gca()

ax.add_patch(
     patches.Polygon(
        xy = points_lst,
        edgecolor = 'blue',
        facecolor = 'blue',
        fill = True,
        alpha = 0.15
    )   
)
plt.show()

# создается директория для ответов
Path("resdir").mkdir(parents=True, exist_ok=True)

# сохранение в rle 
print("Сохранить rle файл: (Y/N)")
ch = input()
if (ch == "Y" or ch == "y"):
    mask_rle = rle_code(cropped)
    rle_to_txt(mask_rle, 'resdir/ResultRle.txt')

# сохранение в png 
print("Сохранить png файл : (Y/N)")
ch = input()
if (ch == "Y" or ch == "y"):
    cv2.imwrite('resdir/Result.png', cropped)
    cv2.waitKey(0)