import cv2
import numpy as np
import ezdxf
import os

def image_to_dxf(image_name, dxf_name):
    if not os.path.exists(image_name):
        return

    # Чтение сразу в 1 канал (Grayscale)
    gray_init = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    
    # ЭТАП ОПТИМИЗАЦИИ ПАМЯТИ: Выполняем бинаризацию и эрозию НА НАТИВНОМ разрешении до апскейла
    _, thresh_init = cv2.threshold(gray_init, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel_unify = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    # 5 итераций эрозии на исходном масштабе схлопнут контуры без перегрузки ОЗУ
    eroded_init = cv2.erode(thresh_init, kernel_unify, iterations=5)
    processed_init = cv2.morphologyEx(eroded_init, cv2.MORPH_CLOSE, kernel_unify, iterations=2)
    
    # Масштабирование уже обработанной маски с использованием стабильной бикубической интерполяции
    scale_factor = 8
    width = int(gray_init.shape[1] * scale_factor)
    height = int(gray_init.shape[0] * scale_factor)
    dim = (width, height)
    
    processed = cv2.resize(processed_init, dim, interpolation=cv2.INTER_CUBIC)
    
    contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    for contour in contours:
        if len(contour) > 5:
            approx = cv2.approxPolyDP(contour, 0.5, False)
            
            if len(approx) > 1:
                points = [(int(pt[0][0]), int(height - pt[0][1])) for pt in approx]
                msp.add_lwpolyline(points, dxfattribs={'layer': 'MIN_GAP_OPTIMIZED'})
            
    doc.saveas(dxf_name)

if __name__ == "__main__":
    image_to_dxf('1.jpg', '1.dxf')
