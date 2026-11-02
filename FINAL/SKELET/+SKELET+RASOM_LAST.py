import cv2
import numpy as np
import ezdxf
import os
import gc
from skimage.morphology import skeletonize

def process_tile_skeleton(tile, scale_factor):
    h, w = tile.shape
    dim = (int(w * scale_factor), int(h * scale_factor))
    
    resized = cv2.resize(tile, dim, interpolation=cv2.INTER_LANCZOS4)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel_pre = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_pre)
    
    binary = closed > 0
    skeleton_binary = skeletonize(binary)
    skel = (skeleton_binary * 255).astype(np.uint8)
    
    contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def image_to_dxf(image_name, dxf_name):
    if not os.path.exists(image_name):
        return

    img_info = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    orig_h, orig_w = img_info.shape[:2]
    del img_info

    scale_factor = 8
    total_height = orig_h * scale_factor
    
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    tile_size = 1200
    
    for y in range(0, orig_h, tile_size):
        for x in range(0, orig_w, tile_size):
            y_end = min(y + tile_size, orig_h)
            x_end = min(x + tile_size, orig_w)
            
            img_full = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            tile = img_full[y:y_end, x:x_end]
            del img_full
            
            contours = process_tile_skeleton(tile, scale_factor)
            
            offset_x = x * scale_factor
            offset_y = y * scale_factor
            
            for contour in contours:
                if len(contour) > 15:
                    # ОПТИМИЗАЦИЯ 1: Увеличен допуск до 1.5 для радикального сокращения узлов и выпрямления линий
                    approx = cv2.approxPolyDP(contour, 1.5, False)
                    if len(approx) > 1:
                        # ОПТИМИЗАЦИЯ 2: Перенос геометрии в виде отдельных простых ОТРЕЗКОВ (LINE) вместо полилиний
                        for i in range(len(approx) - 1):
                            pt1 = approx[i][0]
                            pt2 = approx[i+1][0]
                            
                            start_pt = (int(pt1[0] + offset_x), int(total_height - (pt1[1] + offset_y)))
                            end_pt = (int(pt2[0] + offset_x), int(total_height - (pt2[1] + offset_y)))
                            
                            msp.add_line(start_pt, end_pt, dxfattribs={'layer': 'EDITABLE_AXIS'})
            
            del tile
            del contours
            gc.collect()
            
    doc.saveas(dxf_name)

if __name__ == "__main__":
    image_to_dxf('1.jpg', '1.dxf')
