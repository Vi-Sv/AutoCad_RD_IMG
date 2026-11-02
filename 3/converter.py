import cv2
import numpy as np
import ezdxf
import os

def image_to_dxf(image_name, dxf_name):
    if not os.path.exists(image_name):
        return

    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Чувствительный порог для сохранения мелких деталей и текста
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # RETR_LIST и CHAIN_APPROX_NONE сохраняют абсолютно все точки контура без упрощения
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    h, _, _ = img.shape
    
    for contour in contours:
        if len(contour) > 1:
            # Минимальная аппроксимация для сохранения максимальной детализации
            epsilon = 0.0002 * cv2.arcLength(contour, False)
            approx = cv2.approxPolyDP(contour, epsilon, False)
            
            if len(approx) > 1:
                points = [(int(pt[0][0]), int(h - pt[0][1])) for pt in approx]
                msp.add_lwpolyline(points, dxfattribs={'layer': 'HIGH_DETAIL'})
            
    doc.saveas(dxf_name)

if __name__ == "__main__":
    image_to_dxf('1.jpg', '1.dxf')
