import cv2
import numpy as np
import ezdxf
import os

def image_to_dxf(image_name, dxf_name):
    if not os.path.exists(image_name):
        return

    img = cv2.imread(image_name)
    
    scale_factor = 8
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Использование крестообразного ядра для сохранения углов при агрессивном сжатии
    kernel_unify = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    # Увеличенное до 5 количество итераций эрозии для максимального сближения внутренних стенок контура
    eroded = cv2.erode(thresh, kernel_unify, iterations=5)
    
    # Финальное закрытие микроразрывов
    processed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_unify, iterations=2)
    
    contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    for contour in contours:
        if len(contour) > 20:
            points = [(int(pt[0][0]), int(height - pt[0][1])) for pt in contour]
            msp.add_lwpolyline(points, dxfattribs={'layer': 'MIN_GAP_DETAIL'})
            
    doc.saveas(dxf_name)

if __name__ == "__main__":
    image_to_dxf('1.jpg', '1.dxf')
