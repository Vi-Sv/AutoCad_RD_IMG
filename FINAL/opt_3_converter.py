import cv2
import numpy as np
import ezdxf
import os
import gc

def process_tile(tile, scale_factor):
    """Обработка отдельного небольшого фрагмента изображения."""
    # Бинаризация Оцу
    _, thresh = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Морфология (сжатие двойного контура)
    kernel_unify = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = cv2.erode(thresh, kernel_unify, iterations=5)
    processed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_unify, iterations=2)
    
    # Локальный апскейл фрагмента
    h, w = processed.shape
    dim = (int(w * scale_factor), int(h * scale_factor))
    resized = cv2.resize(processed, dim, interpolation=cv2.INTER_CUBIC)
    
    # Поиск контуров
    contours, _ = cv2.findContours(resized, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def image_to_dxf(image_name, dxf_name):
    if not os.path.exists(image_name):
        return

    # Чтение метаданных изображения без загрузки пикселей в память
    img_info = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    orig_h, orig_w = img_info.shape[:2]
    del img_info

    # Базовые параметры масштабирования
    scale_factor = 8
    total_height = orig_h * scale_factor
    
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # ОПТИМИЗАЦИЯ: Определение безопасного размера тайла (максимум 2000x2000 пикселей за раз)
    tile_size = 2000
    
    # Поблочное сканирование изображения по сетке X и Y
    for y in range(0, orig_h, tile_size):
        for x in range(0, orig_w, tile_size):
            # Расчет точных границ текущего блока с учетом краев
            y_end = min(y + tile_size, orig_h)
            x_end = min(x + tile_size, orig_w)
            
            # Чтение из файла ТОЛЬКО текущего небольшого фрагмента чертежа
            img_full = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            tile = img_full[y:y_end, x:x_end]
            del img_full
            
            # Обработка выделенного блока
            contours = process_tile(tile, scale_factor)
            
            # Смещение координат векторов в общую систему координат DXF
            offset_x = x * scale_factor
            offset_y = y * scale_factor
            
            for contour in contours:
                if len(contour) > 5:
                    approx = cv2.approxPolyDP(contour, 0.5, False)
                    if len(approx) > 1:
                        # Пересчет точек с учетом глобального сдвига тайла и инверсии Y
                        points = [(int(pt[0][0] + offset_x), int(total_height - (pt[0][1] + offset_y))) for pt in approx]
                        msp.add_lwpolyline(points, dxfattribs={'layer': 'MIN_GAP_TILED'})
            
            # Принудительное удаление объектов из памяти и очистка ОЗУ после каждого тайла
            del tile
            del contours
            gc.collect()
            
    doc.saveas(dxf_name)

if __name__ == "__main__":
    image_to_dxf('1.jpg', '1.dxf')
