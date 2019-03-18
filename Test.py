import cv2
import numpy as np

original_img = cv2.imread("F:/0__1.png")
th=cv2.resize(original_img,(2,7))
out=th.reshape(-1,14).astype(np.float32)
print(out)
cv2.imshow("gray", th)    #255白

cv2.waitKey(0)