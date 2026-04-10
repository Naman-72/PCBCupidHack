"""
Breadboard Hole Detector — OpenCV Pipeline
==========================================
Detects the grid of holes on a breadboard and maps them to
standard indices like A1, B12, etc.

Pipeline:
  1. Preprocess  → grayscale + contrast enhancement
  2. Threshold   → isolate dark holes from lighter board
  3. Morphology  → clean up noise, fill partial circles
  4. Blob / Hough Circle detection → find hole centers
  5. Grid mapping → snap centers to a regular grid
  6. Label output → assign row letters + column numbers

Usage:
  python breadboard_hole_detector.py --image breadboard.jpg
  python breadboard_hole_detector.py --image breadboard.jpg --debug
"""

import cv2
import numpy as np
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict


# ──────────────────────────────────────────────
# 1. IMAGE LOADING & PREPROCESSING
# ──────────────────────────────────────────────

def load_and_preprocess(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load image and return both the original BGR image and a preprocessed
    grayscale version suitable for hole detection.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE: boosts local contrast so holes pop even on uneven-lit boards
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Slight blur to reduce sensor noise before thresholding
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    return img, blurred


# ──────────────────────────────────────────────
# 2. THRESHOLDING — isolate holes
# ──────────────────────────────────────────────

def threshold_holes(gray: np.ndarray) -> np.ndarray:
    """
    Breadboard holes are darker than the board surface.
    Adaptive threshold handles uneven lighting across the board.
    """
    # Adaptive threshold: each region is thresholded relative to its local mean.
    # blockSize=21 works well for typical hole sizes; tune if needed.
    thresh = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,   # holes become WHITE
        blockSize=21,
        C=4
    )
    return thresh


# ──────────────────────────────────────────────
# 3. MORPHOLOGICAL CLEANUP
# ──────────────────────────────────────────────

def morphological_cleanup(thresh: np.ndarray, hole_radius_px: int = 6) -> np.ndarray:
    """
    Remove noise and fill partial circles.

    hole_radius_px: approximate radius of one hole in pixels.
                    Adjust based on your image resolution.
    """
    kernel_size = max(3, hole_radius_px // 2 * 2 - 1)   # must be odd
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    # Opening: erode then dilate — kills tiny noise blobs
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Closing: dilate then erode — fills gaps inside holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closed


# ──────────────────────────────────────────────
# 4A. BLOB DETECTION (SimpleBlobDetector)
# ──────────────────────────────────────────────

def detect_blobs(processed: np.ndarray,
                 min_area: int = 50,
                 max_area: int = 2000) -> List[Tuple[int, int]]:
    """
    Use SimpleBlobDetector to find circular blobs = holes.
    Returns list of (x, y) center coordinates.
    """
    params = cv2.SimpleBlobDetector_Params()

    # Filter by area
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area

    # Filter by circularity — holes are (nearly) circular
    params.filterByCircularity = True
    params.minCircularity = 0.5   # 1.0 = perfect circle; lower = more lenient

    # Filter by convexity
    params.filterByConvexity = True
    params.minConvexity = 0.7

    # Filter by inertia (roundness)
    params.filterByInertia = True
    params.minInertiaRatio = 0.4

    params.filterByColor = True
    params.blobColor = 255   # detect white blobs on black background

    detector = cv2.SimpleBlobDetector_create(params)

    # SimpleBlobDetector expects an 8-bit single-channel image
    # We invert because it detects dark blobs by default (unless blobColor=255)
    keypoints = detector.detect(processed)

    centers = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
    return centers


# ──────────────────────────────────────────────
# 4B. HOUGH CIRCLE DETECTION (alternative / complement)
# ──────────────────────────────────────────────

def detect_hough_circles(gray: np.ndarray,
                          min_radius: int = 4,
                          max_radius: int = 15) -> List[Tuple[int, int]]:
    """
    HoughCircles is excellent for nearly-circular holes.
    Works on the blurred grayscale (not the threshold).

    min_radius / max_radius: tune to your image resolution.
    """
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,            # inverse resolution ratio (1 = same resolution)
        minDist=12,      # minimum distance between hole centers (pixels)
        param1=60,       # upper threshold for Canny edge detector
        param2=20,       # accumulator threshold — lower = more circles detected
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype(int)
    centers = [(c[0], c[1]) for c in circles]
    return centers


# ──────────────────────────────────────────────
# 4C. CONTOUR-BASED DETECTION (robust fallback)
# ──────────────────────────────────────────────

def detect_contour_holes(thresh: np.ndarray,
                          min_area: int = 40,
                          max_area: int = 2000,
                          min_circularity: float = 0.45) -> List[Tuple[int, int]]:
    """
    Find contours on the threshold image, filter by area and circularity.
    Very robust — works even when Hough/Blob methods struggle.
    """
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < min_circularity:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    return centers


# ──────────────────────────────────────────────
# 5. MERGE DETECTIONS & DEDUPLICATE
# ──────────────────────────────────────────────

def merge_and_deduplicate(centers_list: List[List[Tuple[int, int]]],
                           min_dist: int = 10) -> List[Tuple[int, int]]:
    """
    Merge centers from multiple detectors and remove duplicates
    that are within min_dist pixels of each other.
    """
    all_centers = []
    for centers in centers_list:
        all_centers.extend(centers)

    if not all_centers:
        return []

    # Sort by x then y for deterministic output
    all_centers.sort(key=lambda p: (p[0], p[1]))

    unique = []
    used = [False] * len(all_centers)

    for i, c in enumerate(all_centers):
        if used[i]:
            continue
        cluster = [c]
        for j in range(i + 1, len(all_centers)):
            if used[j]:
                continue
            dx = all_centers[j][0] - c[0]
            dy = all_centers[j][1] - c[1]
            if np.sqrt(dx * dx + dy * dy) < min_dist:
                cluster.append(all_centers[j])
                used[j] = True
        # Use centroid of the cluster
        cx = int(np.mean([p[0] for p in cluster]))
        cy = int(np.mean([p[1] for p in cluster]))
        unique.append((cx, cy))

    return unique


# ──────────────────────────────────────────────
# 6. GRID MAPPING — snap holes to a regular grid
# ──────────────────────────────────────────────

def cluster_1d(values: List[float], tolerance: int) -> List[float]:
    """
    Group close values (rows or columns) and return sorted cluster centers.
    """
    if not values:
        return []
    sorted_vals = sorted(values)
    clusters = []
    current = [sorted_vals[0]]

    for v in sorted_vals[1:]:
        if v - current[-1] <= tolerance:
            current.append(v)
        else:
            clusters.append(float(np.mean(current)))
            current = [v]
    clusters.append(float(np.mean(current)))
    return clusters


def snap_to_grid(centers: List[Tuple[int, int]],
                 tolerance: int = 8) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Snap detected hole centers to the nearest grid intersection.

    Returns a dict: grid_position (row_idx, col_idx) → pixel center (x, y)
    """
    if not centers:
        return {}

    ys = [c[1] for c in centers]
    xs = [c[0] for c in centers]

    row_centers = cluster_1d(ys, tolerance)
    col_centers = cluster_1d(xs, tolerance)

    grid = {}
    for (x, y) in centers:
        row_idx = int(np.argmin([abs(y - rc) for rc in row_centers]))
        col_idx = int(np.argmin([abs(x - cc) for cc in col_centers]))
        key = (row_idx, col_idx)
        # If two holes map to the same grid cell, keep the closer one
        if key not in grid:
            grid[key] = (x, y)
        else:
            prev = grid[key]
            prev_dist = (prev[0] - col_centers[col_idx])**2 + (prev[1] - row_centers[row_idx])**2
            new_dist  = (x     - col_centers[col_idx])**2 + (y      - row_centers[row_idx])**2
            if new_dist < prev_dist:
                grid[key] = (x, y)

    return grid


# ──────────────────────────────────────────────
# 7. LABEL ASSIGNMENT — row letters + col numbers
# ──────────────────────────────────────────────

def assign_labels(grid: Dict[Tuple[int, int], Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    """
    Convert (row_idx, col_idx) → breadboard label like "A1", "B12", etc.

    Rows → letters (A, B, C, …)
    Columns → 1-based numbers

    Standard 830-point breadboard layout:
      - Rows a–e  (top half) and f–j (bottom half) with a gap in the middle
      - Columns 1–63 (or 1–30 for half-size)
    """
    labeled = {}
    for (row_idx, col_idx), pixel in grid.items():
        row_letter = chr(ord('A') + row_idx)   # A=0, B=1, …
        col_number = col_idx + 1               # 1-indexed
        label = f"{row_letter}{col_number}"
        labeled[label] = pixel
    return labeled


# ──────────────────────────────────────────────
# 8. VISUALIZATION
# ──────────────────────────────────────────────

def draw_detections(img: np.ndarray,
                    labeled: Dict[str, Tuple[int, int]],
                    show_labels: bool = True) -> np.ndarray:
    """
    Draw circles and labels on the original image.
    """
    vis = img.copy()

    for label, (x, y) in labeled.items():
        # Draw hole circle
        cv2.circle(vis, (x, y), 6, (0, 255, 0), 2)
        # Draw center dot
        cv2.circle(vis, (x, y), 2, (0, 200, 255), -1)

        if show_labels:
            # Only label every other hole to avoid clutter
            row_letter = label[0]
            col_num = int(label[1:])
            if col_num % 5 == 1:   
                cv2.putText(
                    vis, label, (x - 8, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (255, 255, 0), 1, cv2.LINE_AA
                )

    return vis


def draw_debug_stages(gray_blurred: np.ndarray,
                      thresh: np.ndarray,
                      cleaned: np.ndarray) -> np.ndarray:
    """
    Side-by-side view: blurred gray | threshold | cleaned morphology.
    """
    h, w = gray_blurred.shape
    blurred_bgr = cv2.cvtColor(gray_blurred, cv2.COLOR_GRAY2BGR)
    thresh_bgr  = cv2.cvtColor(thresh,       cv2.COLOR_GRAY2BGR)
    cleaned_bgr = cv2.cvtColor(cleaned,      cv2.COLOR_GRAY2BGR)

    def label_img(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return out

    panel = np.hstack([
        label_img(blurred_bgr, "1. Preprocessed"),
        label_img(thresh_bgr,  "2. Threshold"),
        label_img(cleaned_bgr, "3. Morphology")
    ])
    return panel


# ──────────────────────────────────────────────
# 9. MAIN PIPELINE
# ──────────────────────────────────────────────

def detect_breadboard_holes(image_path: str,
                             debug: bool = False,
                             min_radius: int = 4,
                             max_radius: int = 15,
                             grid_tolerance: int = 8) -> Dict[str, Tuple[int, int]]:
    """
    Full pipeline. Returns dict of label → (x, y) pixel position.
    """
    print(f"[1/6] Loading image: {image_path}")
    img, gray_blurred = load_and_preprocess(image_path)

    print("[2/6] Thresholding …")
    thresh = threshold_holes(gray_blurred)

    print("[3/6] Morphological cleanup …")
    cleaned = morphological_cleanup(thresh, hole_radius_px=max_radius)

    print("[4/6] Detecting holes (3 methods) …")
    blob_centers   = detect_blobs(cleaned)
    hough_centers  = detect_hough_circles(gray_blurred, min_radius, max_radius)
    contour_centers = detect_contour_holes(cleaned)

    print(f"       Blobs: {len(blob_centers)}  |  "
          f"Hough: {len(hough_centers)}  |  "
          f"Contours: {len(contour_centers)}")

    print("[5/6] Merging & deduplicating …")
    all_centers = merge_and_deduplicate(
        [blob_centers, hough_centers, contour_centers],
        min_dist=max_radius
    )
    print(f"       Unique holes detected: {len(all_centers)}")

    print("[6/6] Mapping to grid …")
    grid   = snap_to_grid(all_centers, tolerance=grid_tolerance)
    labeled = assign_labels(grid)
    print(f"       Grid size: {len(labeled)} labeled holes")

    # ── Visualize ──
    vis = draw_detections(img, labeled)
    cv2.imwrite("output_detected.jpg", vis)
    print("\n✅ Saved annotated image → output_detected.jpg")

    if debug:
        debug_panel = draw_debug_stages(gray_blurred, thresh, cleaned)
        cv2.imwrite("output_debug_stages.jpg", debug_panel)
        print("✅ Saved debug stages → output_debug_stages.jpg")

    # Print the label map
    print("\n── Hole Map (label → pixel center) ──")
    for label in sorted(labeled.keys(), key=lambda s: (s[0], int(s[1:]))):
        x, y = labeled[label]
        print(f"  {label:>5}  →  pixel ({x:4d}, {y:4d})")

    return labeled


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breadboard Hole Detector")
    parser.add_argument("--image",  required=True, help="Path to breadboard image")
    parser.add_argument("--debug",  action="store_true", help="Save debug stage images")
    parser.add_argument("--min-radius", type=int, default=4,  help="Min hole radius in pixels (default 4)")
    parser.add_argument("--max-radius", type=int, default=15, help="Max hole radius in pixels (default 15)")
    parser.add_argument("--grid-tolerance", type=int, default=8,
                        help="Pixel tolerance when snapping to grid rows/cols (default 8)")
    args = parser.parse_args()

    labeled_holes = detect_breadboard_holes(
        image_path=args.image,
        debug=args.debug,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        grid_tolerance=args.grid_tolerance
    )

    print(f"\nTotal holes mapped: {len(labeled_holes)}")