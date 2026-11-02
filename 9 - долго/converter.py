import cv2
import numpy as np
import ezdxf
import os

def zhang_suen_skeletonize(img):
    # Реализация классического алгоритма Зонга-Суиня для сохранения связности тонких линий
    skel = img.copy()
    done = False
    while not done:
        done = True
        for pass_num in (1, 2):
            # Поиск граничных пикселей и их соседей
            h, w = skel.shape
            mask = np.zeros_like(skel)
            # Вычисление структуры соседей для каждого пикселя
            for y in range(1, h-1):
                for x in range(1, w-1):
                    if skel[y, x] == 0:
                        continue
                    p2, p3, p4 = skel[y-1, x], skel[y-1, x+1], skel[y, x+1]
                    p5, p6, p7 = skel[y+1, x+1], skel[y+1, x], skel[y+1, x-1]
                    p8, p9 = skel[y, x-1], skel[y-1, x-1]
                    
                    # Нормализация к 0 и 1
                    p2, p3, p4, p5, p6, p7, p8, p9 = [int(p == 255) for p in [p2, p3, p4, p5, p6, p7, p8, p9]]
                    
                    b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                    if not (2 <= b <= 6):
                        continue
                        
                    # Подсчет переходов 0 -> 1
                    seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
                    a = sum((seq[i] == 0 and seq[i+1] == 1) for i in range(8))
                    if a != 1:
                        continue
                        
                    if pass_num == 1:
                        if (p2 * p4 * p6 == 0) and (p4 * p6 * p8 == 0):
                            mask[y, x] = 255
                            done = False
                    else:
                        if (p2 * p4 * p8 == 0) and (p2 * p6 * p8 == 0):
                            mask[y, x] = 255
                            done = False
            skel[mask == 255] = 0
    return skel

def image_to_dxf(image_name, dxf_name):
    if not os.path.exists(image_name):
        return

    img = cv2.imread(image_name)
    
    scale_factor = 8
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Морфологическое предзаполнение микропор перед утончением
    kernel_pre = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_pre)
    
    # Идеальное схлопывание двойного контура в одну центральную нить
    skel = zhang_suen_skeletonize(closed)
    
    contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    for contour in contours:
        if len(contour) > 15:
            points = [(int(pt[0][0]), int(height - pt[0][1])) for pt in contour]
            msp.add_lwpolyline(points, dxfattribs={'layer': 'SINGLE_AXIS_DETAIL'})
            
    doc.saveas(dxf_name)

if __name__ == "__main__":
    image_to_dxf('1.jpg', '1.dxf')
