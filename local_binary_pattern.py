import numpy as np

def get_pixel(img, center, x, y):
    new_value = 0
      
    try:
        if img[x][y] >= center:
            new_value = 1          
    except:
        pass
      
    return new_value

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []

    val_ar.append(get_pixel(img, center, x-1, y-1))
    val_ar.append(get_pixel(img, center, x-1, y))
    val_ar.append(get_pixel(img, center, x-1, y + 1))
    val_ar.append(get_pixel(img, center, x, y + 1))
    val_ar.append(get_pixel(img, center, x + 1, y + 1))
    val_ar.append(get_pixel(img, center, x + 1, y))
    val_ar.append(get_pixel(img, center, x + 1, y-1))
    val_ar.append(get_pixel(img, center, x, y-1))
       
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
      
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
          
    return val

def LBP(img_gray, image_shape):
	h, w = image_shape[:2]

	img_lbp = np.zeros((h, w), np.uint8)

	for i in range(0, h):
	    for j in range(0, w):
	        img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

	return img_lbp