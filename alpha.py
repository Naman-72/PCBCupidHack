import cv2
import numpy as np

img = cv2.imread("hungry.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

th = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    15, 3
)

# emphasize long horizontal and vertical structures
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))

horizontal = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_h)
vertical = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_v)

mask = cv2.bitwise_or(horizontal, vertical)

lines = cv2.HoughLinesP(
    mask,
    rho=1,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=80,
    maxLineGap=15
)

out = img.copy()
if lines is not None:
    for l in lines:
        x1, y1, x2, y2 = l[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        if abs(angle) < 10 or abs(abs(angle) - 90) < 10:
            cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite("detected_lines.jpg", out)