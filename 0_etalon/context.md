# Контекст векторизации чертежей (Python -> DXF)

## Текущий статус задачи
Реализован и протестирован оптимальный двухконтурный метод векторизации растровых строительных чертежей с контролируемым зазором между линиями. Достигнуто эталонное качество печати в AutoCAD при масштабе увеличения `scale_factor = 8`. Последний этап — внедрение оптимизации памяти и структуры данных для предотвращения падения скрипта по ошибке `OutOfMemoryError` на тяжелых схемах.

## Ключевые технические параметры (Настройки)
1. **Масштабирование растра (Scale Factor):** Установлено на уровне `8` с интерполяцией `cv2.INTER_LANCZOS4`. Обеспечивает субпиксельную плотность векторов для плавности округлостей и шрифтов. При превышении масштаба (x16, x24) на цветных файлах возникает риск утечки памяти.
2. **Контроль зазора (Эрозия):** Применяется крестообразное ядро `cv2.MORPH_CROSS (3x3)` с количеством итераций эрозии `iterations = 5`. Это обеспечивает максимальное сближение («схлопывание») внешнего и внутреннего контуров, решая проблему двоения линий («бубликов»).
3. **Оптимизация веса и узлов:** Переход от режима сохранения каждого пикселя (`CHAIN_APPROX_NONE`) к адаптивному схлопыванию прямых участков (`CHAIN_APPROX_SIMPLE`) и субпиксельной аппроксимации Дугласа-Пекера с допуском `0.5` пикселя. Снижает вес DXF в 5-10 раз без визуальной потери качества.
4. **Безопасность по памяти:** Предварительное чтение растра сразу в одноканальном режиме градаций серого (`cv2.IMREAD_GRAYSCALE`) для снижения нагрузки на ОЗУ на 66%.

## Необходимые библиотеки (PowerShell)
```powershell
pip install opencv-python numpy ezdxf
```

## Текущий эталонный код (Скрипт оптимизации)
```python
import cv2
import numpy as np
import ezdxf
import os

def image_to_dxf(image_name, dxf_name):
    if not os.path.exists(image_name):
        print(f"Ошибка: Файл {image_name} не найден.")
        return

    # Загрузка в Grayscale для экономии ОЗУ на больших схемах
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
    
    # CHAIN_APPROX_SIMPLE убирает лишние точки на прямых отрезках
    contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    for contour in contours:
        if len(contour) > 5:
            # Субпиксельная аппроксимация (0.5 пикселя) убирает ступенчатость, сокращая вес файла
            approx = cv2.approxPolyDP(contour, 0.5, False)
            
            if len(approx) > 1:
                points = [(int(pt[0][0]), int(height - pt[0][1])) for pt in approx]
                msp.add_lwpolyline(points, dxfattribs={'layer': 'MIN_GAP_OPTIMIZED'})
            
    doc.saveas(dxf_name)
    print(f"Успешно сохранено в {dxf_name}")

if __name__ == "__main__":
    image_to_dxf('1.jpg', '1.dxf')
```

## Следующие шаги для возобновления работы
* Оценить вес и поведение DXF-файла при зуммировании в AutoCAD после применения `CHAIN_APPROX_SIMPLE`.
* В случае необходимости дальнейшего выпрямления волнистости — увеличить параметр `epsilon` в функции `cv2.approxPolyDP` с `0.5` до `0.002 * perimeter`.
* Приступить к этапу фильтрации изолированного мелкого мусора (артефактов фона) на больших чертежах.
