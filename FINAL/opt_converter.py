import cv2
import numpy as np
import ezdxf
import os

def image_to_dxf(image_name, dxf_name):
    if not os.path.exists(image_name):
        return

    # Загрузка сразу в Grayscale экономит память при обработке больших схем
    gray_init = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    
    scale_factor = 8
    width = int(gray_init.shape[1] * scale_factor)
    height = int(gray_init.shape[0] * scale_factor)
    dim = (width, height)
    
    resized = cv2.resize(gray_init, dim, interpolation=cv2.INTER_LANCZOS4)
    
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel_unify = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    eroded = cv2.erode(thresh, kernel_unify, iterations=5)
    processed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_unify, iterations=2)
    
    # ОПТИМИЗАЦИЯ 1: CHAIN_APPROX_SIMPLE вместо NONE автоматически убирает лишние точки на прямых линиях
    contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    for contour in contours:
        if len(contour) > 5: # Порог уменьшен, так как точек в контурах стало физически меньше
            # ОПТИМИЗАЦИЯ 2: Субпиксельная аппроксимация (0.5 пикселя) убирает ступенчатость, сокращая вес файла
            approx = cv2.approxPolyDP(contour, 0.5, False)
            
            if len(approx) > 1:
                points = [(int(pt[0][0]), int(height - pt[0][1])) for pt in approx]
                msp.add_lwpolyline(points, dxfattribs={'layer': 'MIN_GAP_OPTIMIZED'})
            
    doc.saveas(dxf_name)

if __name__ == "__main__":
    image_to_dxf('1.jpg', '1.dxf')
