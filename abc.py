import cv2
import numpy as np
from math import pi

# ---------------------------------------------------
# helpers
# ---------------------------------------------------
def circularity(contour):
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return 0
    return 4 * pi * area / (peri * peri)

def cluster_1d_points(values, tol=8):
    """
    Cluster 1D points.
    Returns list of clusters, where each cluster is a list of values.
    """
    values = sorted(values)
    if not values:
        return []

    clusters = [[values[0]]]
    for v in values[1:]:
        if abs(v - np.mean(clusters[-1])) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return clusters

def cluster_centers_with_counts(values, tol=8):
    clusters = cluster_1d_points(values, tol)
    result = []
    for c in clusters:
        result.append({
            "center": int(round(np.mean(c))),
            "count": len(c),
            "values": c
        })
    return result

# ---------------------------------------------------
# Step 1: load image
# ---------------------------------------------------
img = cv2.imread("breadboard.jpg")
if img is None:
    raise RuntimeError("image not found")

vis = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------
# Step 2: detect breadboard ROI
# ---------------------------------------------------
# Breadboard is a large light-gray object, so threshold bright-ish areas
# then keep the biggest meaningful contour.
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# light objects
_, board_mask = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)

# close gaps a little
kernel = np.ones((9, 9), np.uint8)
board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

board_box = None
max_area = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 100000:
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    # breadboard is wide rectangle
    if w < 600 or h < 300:
        continue
    if area > max_area:
        max_area = area
        board_box = (x, y, w, h)

if board_box is None:
    raise RuntimeError("Could not find breadboard ROI")

bx, by, bw, bh = board_box

# shrink ROI a little so outer background/text is excluded
pad_x = int(0.03 * bw)
pad_y = int(0.05 * bh)

rx1 = bx + pad_x
ry1 = by + pad_y
rx2 = bx + bw - pad_x
ry2 = by + bh - pad_y

roi = img[ry1:ry2, rx1:rx2].copy()
roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

# ---------------------------------------------------
# Step 3: detect hole candidates only inside ROI
# ---------------------------------------------------
roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)

# holes are dark
_, th = cv2.threshold(roi_blur, 180, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((3, 3), np.uint8)
th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

hole_centers = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 15 or area > 200:
        continue

    x, y, w, h = cv2.boundingRect(cnt)
    ar = w / float(h)

    if ar < 0.6 or ar > 1.4:
        continue

    circ = circularity(cnt)
    if circ < 0.35:
        continue

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # convert ROI coordinates back to full image coordinates
    gx = rx1 + cx
    gy = ry1 + cy
    hole_centers.append((gx, gy))

# ---------------------------------------------------
# Step 4: cluster x and y
# ---------------------------------------------------
xs = [p[0] for p in hole_centers]
ys = [p[1] for p in hole_centers]

col_clusters = cluster_centers_with_counts(xs, tol=10)
row_clusters = cluster_centers_with_counts(ys, tol=10)

# IMPORTANT:
# keep only clusters that have enough detections
# fake rows/cols outside board usually have tiny support
min_points_per_row = 8
min_points_per_col = 4

col_clusters = [c for c in col_clusters if c["count"] >= min_points_per_col]
row_clusters = [c for c in row_clusters if c["count"] >= min_points_per_row]

col_centers = [c["center"] for c in col_clusters]
row_centers = [c["center"] for c in row_clusters]

print("Detected holes:", len(hole_centers))
print("Detected vertical lines:", len(col_centers))
print("Detected horizontal lines:", len(row_centers))

print("Row centers:", row_centers)
print("Column centers:", col_centers)

# ---------------------------------------------------
# Step 5: draw results
# ---------------------------------------------------
for (cx, cy) in hole_centers:
    cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)

for x in col_centers:
    cv2.line(vis, (x, ry1), (x, ry2), (255, 0, 0), 1)

for y in row_centers:
    cv2.line(vis, (rx1, y), (rx2, y), (0, 255, 0), 1)

cv2.imwrite("board_roi_and_grid.png", vis)
print("Saved: board_roi_and_grid.png")