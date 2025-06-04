import numpy as np
import cv2
import math
from math import cos, sin, radians

def preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    H, W = blurred.shape
    if H != W :
        l = math.ceil(math.log2(max(H, W)))
        size = 2**l
        canvas = np.full((size, size), fill_value=-1, dtype=np.float32)
        canvas[0:H, 0:W] = blurred
    else :
        l = int(math.log2(H))
        canvas = blurred

    return canvas, image_rgb, lab, gray

def integral_image(image) :
    return cv2.integral(image, sdepth = cv2.CV_64F)

def hitung_sdmap(canvas, R) :
    image = canvas.astype(np.float32)
    image_kuadrat = image**2
    mask = (canvas >= 0).astype(np.float32)

    integral = integral_image(image * mask)
    integral_kuadrat = integral_image(image_kuadrat * mask)
    integral_mask = integral_image(mask)

    H, W = image.shape
    sd_map = np.zeros((H, W), dtype = np.float32)

    for y in range(H) :
        for x in range(W) :
            x1 = max(x - R, 0)
            y1 = max(y - R, 0)
            x2 = min(x + R, W - 1)
            y2 = min(y + R, H - 1)

            A = (y1, x1)
            B = (y1, x2 + 1)
            C = (y2 + 1, x1)
            D = (y2 + 1, x2 + 1)
            area = (y2 - y1 + 1) * (x2 - x1 + 1)

            sum_ = integral[D] - integral[B] - integral[C] + integral[A]
            sum_kuadrat = integral_kuadrat[D] - integral_kuadrat[B] - integral_kuadrat[C] + integral_kuadrat[A]
            count = integral_mask[D] - integral_mask[B] - integral_mask[C] + integral_mask[A]

            if count > 0:
                mean = sum_ / count
                mean_kuadrat = sum_kuadrat / count
                var = mean_kuadrat - (mean ** 2)
                std = np.sqrt(var) if var > 0 else 0
                sd_map[y, x] = std
            else:
                sd_map[y, x] = -1

    sd_map[canvas < 0] = -1
    sd_map_norm = cv2.normalize(sd_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return sd_map, sd_map_norm

def is_homogen(img, y, x, h, w, split_threshold) :
    segment = img[y : y + h, x : x + w]
    if np.any(segment < 0) :
        return False
    return np.var(segment) <= split_threshold

def split(img, y, x, h, w, min_size, split_threshold, depth=0) :
    if w <= min_size or h <= min_size or is_homogen(img, y, x, h, w, split_threshold) :
        return [(x, y, w, h, depth)]
    else :
        mid_w = w // 2
        mid_h = h // 2
        blok = []

        blok += split(img, y, x, mid_h, mid_w, min_size, split_threshold, depth+1)
        blok += split(img, y, x + mid_w, mid_h, mid_w, min_size, split_threshold, depth+1)
        blok += split(img, y + mid_h, x, h - mid_h, mid_w, min_size, split_threshold, depth+1)
        blok += split(img, y + mid_h, x + mid_w, h - mid_h, w - mid_w, min_size, split_threshold, depth+1)
        return blok
    
def hitung_glcm(gray, level, distances=[1], angles=[0, 45, 90, 135]):
    glcm_matrix = {}
    max_val = gray.max()
    quantized = (gray / max_val * (level - 1)).astype(np.uint8)

    for d in distances:
        for angle in angles:
            matrix = np.zeros((level, level), dtype=np.uint32)
            dx = int(round(d * cos(radians(angle))))
            dy = int(round(-d * sin(radians(angle))))
            rows, cols = quantized.shape
            for y in range(rows):
                for x in range(cols):
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < rows and 0 <= nx < cols:
                        i = quantized[y, x]
                        j = quantized[ny, nx]
                        matrix[i, j] += 1
            symmetric_matrix = matrix + matrix.T
            glcm_matrix[(d, angle)] = symmetric_matrix
          
    return glcm_matrix

def normalisasi_glcm(glcm_matrix):
    total = np.sum(glcm_matrix)
    if total == 0:
        return glcm_matrix.astype(np.float64)
    return glcm_matrix.astype(np.float64) / total

def ekstrak_glcm(glcm_matrix):
    total_contrast = 0
    total_energy = 0
    total_homogeneity = 0
    jumlah = 0

    for matrix in glcm_matrix.values():
        norm_matrix = normalisasi_glcm(matrix)
        levels = norm_matrix.shape[0]
        i = np.arange(levels)
        j = np.arange(levels)
        I, J = np.meshgrid(i, j, indexing='ij')

        total_contrast += np.sum(norm_matrix * (I - J) ** 2)
        total_energy += np.sum(norm_matrix ** 2)
        total_homogeneity += np.sum(norm_matrix / (1.0 + np.abs(I - J)))
        jumlah += 1

    return {
        'contrast': total_contrast / jumlah,
        'energy': total_energy / jumlah,
        'homogeneity': total_homogeneity / jumlah
    }

def check_merge_condition(mean_sd1, mean_sd2, mean_lab1, mean_lab2, sigma_T, color_T, h_threshold, c_threshold, e_threshold, glcm_feat1=None, glcm_feat2=None):
    diff_sd = abs(mean_sd1 - mean_sd2)
    diff_color = np.linalg.norm(mean_lab1 - mean_lab2)

    diff_h = 0
    diff_c = 0
    diff_e = 0
    if glcm_feat1 is not None and glcm_feat2 is not None:
        diff_h = abs(glcm_feat1['homogeneity'] - glcm_feat2['homogeneity'])
        diff_c = abs(glcm_feat1['contrast'] - glcm_feat2['contrast'])
        diff_e = abs(glcm_feat1['energy'] - glcm_feat2['energy'])

    return (diff_sd < sigma_T) and (diff_color < color_T) and \
       (diff_h < h_threshold) and ((diff_c < c_threshold) or (diff_e < e_threshold))

def merge_blocks(blok, sd_map, gray, lab, sigma_T, color_T, h_threshold, c_threshold, e_threshold):
    class_map = np.zeros_like(sd_map, dtype=np.int32)
    blok_dict = {}
    class_id = 1

    blok_sorted = sorted(blok, key=lambda b: (b[1], b[0]))

    for x, y, w, h, _ in blok_sorted:
        region_sd = sd_map[y:y+h, x:x+w]
        region_lab = lab[y:y+h, x:x+w]

        if np.any(region_sd < 0):
            continue

        mean_sd = np.mean(region_sd)
        mean_lab = np.mean(region_lab.reshape(-1, 3), axis=0)

        gray_block = gray[y:y+h, x:x+w]
        glcm_matrix = hitung_glcm(gray_block, 8)
        glcm_features = ekstrak_glcm(glcm_matrix)

        blok_dict[(x, y, w, h)] = (mean_sd, mean_lab, glcm_features)

        found_merge = False
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx * w, y + dy * h
            neighbor_key = (nx, ny, w, h)
            if neighbor_key in blok_dict:
                mean_sd_neighbor, mean_lab_neighbor, glcm_feat_neighbor = blok_dict[neighbor_key]
                if check_merge_condition(mean_sd, mean_sd_neighbor, mean_lab, mean_lab_neighbor, sigma_T, color_T,
                                        h_threshold, c_threshold, e_threshold, glcm_features, glcm_feat_neighbor):
                    neighbor_label = class_map[ny:ny+h, nx:nx+w][0, 0]
                    if neighbor_label > 0:
                        class_map[y:y+h, x:x+w] = neighbor_label
                        found_merge = True
                        break

        if not found_merge:
            region = class_map[y:y+h, x:x+w]
            mask_empty = (region == 0)
            region[mask_empty] = class_id
            class_map[y:y+h, x:x+w] = region
            class_id += 1

    return class_map

def remove_small_classes(class_map, min_size=64):
    output = class_map.copy()
    unique_labels = np.unique(output)
    for label in unique_labels:
        if label == 0:
            continue
        mask = (output == label)
        if np.sum(mask) < min_size:
            output[mask] = 0
    return output

def combine(class_map, lab):
    H, W = class_map.shape
    filled_map = class_map.copy()
    no_classes = np.argwhere(filled_map == 0)
    arah = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]

    for y, x in no_classes:
        tetangga = []
        pixel_lab1 = lab[y, x]
        for dy, dx in arah:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                pixel_lab2 = lab[ny, nx]
                label = filled_map[ny, nx]
                if label > 0:
                    delta = np.linalg.norm(pixel_lab1 - pixel_lab2)
                    tetangga.append((label, delta))

        if tetangga:
            best_label = min(tetangga, key=lambda x: x[1])[0]
            filled_map[y, x] = best_label

    return filled_map

def simplify_to_two_classes(class_map, lab, color_T):
    output_map = np.zeros_like(class_map)
    labels = [l for l in np.unique(class_map) if l != 0]

    class_means = {}
    for label in labels:
        mask = (class_map == label)
        mean_lab = np.mean(lab[mask], axis=0)
        class_means[label] = mean_lab

    ref_label = labels[0]
    ref_color = class_means[ref_label]

    for label in labels:
        color_dist = np.linalg.norm(class_means[label] - ref_color)
        if color_dist < color_T:
            output_map[class_map == label] = 1
        else:
            output_map[class_map == label] = 2

    return output_map

def overlay_mask(image, class_map, alpha=0.5):
    overlay = image.copy()

    color_map = {
        1: (0, 255, 0),
        2: (0, 0, 255),
    }

    for cls, color in color_map.items():
        mask = (class_map == cls)
        overlay[mask] = (np.array(color) * alpha + overlay[mask] * (1 - alpha)).astype(np.uint8)

    return overlay
