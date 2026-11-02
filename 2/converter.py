import cv2
import numpy as np
import ezdxf
import os

def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            done = True
    return skel

def image_to_dxf(image_name, dxf_name):
    if not os.path.exists(image_name):
        return

    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Бинаризация
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    
    # Скелетизация для устранения двойных линий
    skel = skeletonize(thresh)
    
    # Поиск контуров по осевым линиям
    contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    h, _, _ = img.shape
    
    for contour in contours:
        if len(contour) > 1:
            # Аппроксимация для выпрямления волнистых линий и уменьшения узлов
            epsilon = 0.002 * cv2.arcLength(contour, False)
            approx = cv2.approxPolyDP(contour, epsilon, False)
            
            if len(approx) > 1:
                points = [(int(pt[0][0]), int(h - pt[0][1])) for pt in approx]
                msp.add_lwpolyline(points, dxfattribs={'layer': 'AXIS_LINES'})
            
    doc.saveas(dxf_name)

if __name__ == "__main__":
    image_to_dxf('1.jpg', '1.dxf')
