import cv2
import numpy as np
import ezdxf
import os

def image_to_dxf(image_name, dxf_name):
    if not os.path.exists(image_name):
        return

    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Максимально чувствительный порог (минимальный блок 7x2) для захвата каждого пикселя
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)
    
    # CHAIN_APPROX_NONE извлекает абсолютно все граничные пиксели без малейшего упрощения
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    h, _, _ = img.shape
    
    for contour in contours:
        if len(contour) > 1:
            # Полное отключение аппроксимации: перенос каждого пикселя напрямую в узлы полилинии
            points = [(int(pt[0][0]), int(h - pt[0][1])) for pt in contour]
            msp.add_lwpolyline(points, dxfattribs={'layer': 'MAX_DETAIL'})
            
    doc.saveas(dxf_name)

if __name__ == "__main__":
    image_to_dxf('1.jpg', '1.dxf')
