import cv2
import numpy as np
import ezdxf
import os
import gc

def process_tile(tile, scale_factor):
    # Локальный апскейл небольшого фрагмента до субпиксельной плотности (без перегрузки ОЗУ)
    h, w = tile.shape
    dim = (int(w * scale_factor), int(h * scale_factor))
    resized = cv2.resize(tile, dim, interpolation=cv2.INTER_LANCZOS4)
    
    # Бинаризация Оцу на увеличенном фрагменте
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Контролируемая эрозия на высоком разрешении (линии сужаются, но не исчезают)
    kernel_unify = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = cv2.erode(thresh, kernel_unify, iterations=5)
    processed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_unify, iterations=2)
    
    contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def image_to_dxf(image_name, dxf_name):
    if not os.path.exists(image_name):
        return

    # Предварительное быстрое чтение размеров без загрузки пикселей
    img_info = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    orig_h, orig_w = img_info.shape[:2]
    del img_info

    scale_factor = 8
    total_height = orig_h * scale_factor
    
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Фиксированный размер тайла (безопасный буфер для ОЗУ ПК)
    tile_size = 1500
    
    for y in range(0, orig_h, tile_size):
        for x in range(0, orig_w, tile_size):
            y_end = min(y + tile_size, orig_h)
            x_end = min(x + tile_size, orig_w)
            
            # Чтение только активного фрагмента
            img_full = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            tile = img_full[y:y_end, x:x_end]
            del img_full
            
            contours = process_tile(tile, scale_factor)
            
            offset_x = x * scale_factor
            offset_y = y * scale_factor
            
            for contour in contours:
                if len(contour) > 20: # Возвращен рабочий порог фильтрации шума
                    approx = cv2.approxPolyDP(contour, 0.5, False)
                    if len(approx) > 1:
                        points = [(int(pt[0][0] + offset_x), int(total_height - (pt[0][1] + offset_y))) for pt in approx]
                        msp.add_lwpolyline(points, dxfattribs={'layer': 'TILED_MIN_GAP'})
            
            del tile
            del contours
            gc.collect()
            
    doc.saveas(dxf_name)

if __name__ == "__main__":
    image_to_dxf('1.jpg', '1.dxf')
