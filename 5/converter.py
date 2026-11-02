import cv2
import numpy as np
import ezdxf
import os

def image_to_dxf(image_name, dxf_name):
    if not os.path.exists(image_name):
        return

    img = cv2.imread(image_name)
    
    # Искусственное увеличение разрешения в 4 раза (апскейлинг до 4K/2K) для плотности векторов
    scale_factor = 4
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Порог на увеличенном разрешении для захвата субпиксельных деталей
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
    
    # Извлечение каждого пикселя без пропусков
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    for contour in contours:
        if len(contour) > 1:
            # Перенос сверхплотных координат с инверсией Y
            points = [(int(pt[0][0]), int(height - pt[0][1])) for pt in contour]
            msp.add_lwpolyline(points, dxfattribs={'layer': 'ULTRA_2K_DETAIL'})
            
    doc.saveas(dxf_name)

if __name__ == "__main__":
    image_to_dxf('1.jpg', '1.dxf')
