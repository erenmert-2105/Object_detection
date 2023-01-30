import cv2
import pyscreenshot as ImageGrab
import numpy as np

Battle = cv2.imread('Battle.png', cv2.TM_CCOEFF_NORMED)

def ask(arr):
    
    img=ImageGrab.grab(bbox=(213,160,1635,960))
    full = np.array(img)
    
    result = cv2.matchTemplate(full, arr, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val >= 0.7:
        return True
    else:
        return False