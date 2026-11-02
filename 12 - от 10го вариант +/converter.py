import cv2
import numpy as np
import ezdxf
import os
from skimage.morphology import skeletonize

def image_to_dxf(image_name, dxf_name):
    if not os.path.exists(image_name):
        return

    img = cv2.imread(image_name)
    
    # Экстремальное масштабирование x16 для максимальной субпиксельной плотности одиночной оси
    scale_factor = 16
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Размер ядра адаптирован под масштаб x16 для закрытия микропор
    kernel_pre = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_pre)
    
    binary = closed > 0
    skeleton_binary = skeletonize(binary)
    skel = (skeleton_binary * 255).astype(np.uint8)
    
    contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    for contour in contours:
        # Минимальный порог отсечения шума увеличен до 30 из-за масштаба x16
        if len(contour) > 30:
            points = [(int(pt[0][0]), int(height - pt[0][1])) for pt in contour]
            msp.add_lwpolyline(points, dxfattribs={'layer': 'ULTRA_SINGLE_AXIS'})
            
    doc.saveas(dxf_name)

if __name__ == "__main__":
    image_to_dxf('1.jpg', '1.dxf')
