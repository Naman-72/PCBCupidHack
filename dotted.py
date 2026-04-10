import cv2
import numpy as np


def cluster_1d(points, axis=0, tolerance=10):
    """
    Group points by similar x or y coordinate.
    axis=0 => group by x (vertical lines)
    axis=1 => group by y (horizontal lines)
    """
    if not points:
        return []

    pts = sorted(points, key=lambda p: p[axis])
    groups = [[pts[0]]]

    for p in pts[1:]:
        if abs(p[axis] - groups[-1][-1][axis]) <= tolerance:
            groups[-1].append(p)
        else:
            groups.append([p])

    return groups


def detect_dotted_lines_hv(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Dark dots -> white blobs
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours = dot candidates
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5 or area > 200:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # Keep roughly dot-like blobs
        if 0.5 < aspect_ratio < 2.0:
            cx = x + w // 2
            cy = y + h // 2
            centers.append((cx, cy))

            if debug:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 1)

    if len(centers) < 2:
        return output, []

    min_points_in_line = 4
    coord_tolerance = 10   # allowed difference in x or y for grouping
    max_gap = 40           # max allowed gap between neighboring dots

    final_lines = []

    # -------------------------
    # Horizontal dotted lines
    # group by similar y
    # -------------------------
    horizontal_groups = cluster_1d(centers, axis=1, tolerance=coord_tolerance)

    for group in horizontal_groups:
        if len(group) < min_points_in_line:
            continue

        # sort left to right
        group = sorted(group, key=lambda p: p[0])

        # check spacing consistency
        filtered = [group[0]]
        for i in range(1, len(group)):
            if group[i][0] - group[i - 1][0] <= max_gap:
                filtered.append(group[i])

        if len(filtered) >= min_points_in_line:
            y_avg = int(np.mean([p[1] for p in filtered]))
            x1 = min(p[0] for p in filtered)
            x2 = max(p[0] for p in filtered)

            pt1 = (x1, y_avg)
            pt2 = (x2, y_avg)
            final_lines.append(("horizontal", pt1, pt2, filtered))

            cv2.line(output, pt1, pt2, (0, 0, 255), 2)
            for x, y in filtered:
                cv2.circle(output, (x, y), 3, (0, 255, 0), -1)

    # -------------------------
    # Vertical dotted lines
    # group by similar x
    # -------------------------
    vertical_groups = cluster_1d(centers, axis=0, tolerance=coord_tolerance)

    for group in vertical_groups:
        if len(group) < min_points_in_line:
            continue

        # sort top to bottom
        group = sorted(group, key=lambda p: p[1])

        # check spacing consistency
        filtered = [group[0]]
        for i in range(1, len(group)):
            if group[i][1] - group[i - 1][1] <= max_gap:
                filtered.append(group[i])

        if len(filtered) >= min_points_in_line:
            x_avg = int(np.mean([p[0] for p in filtered]))
            y1 = min(p[1] for p in filtered)
            y2 = max(p[1] for p in filtered)

            pt1 = (x_avg, y1)
            pt2 = (x_avg, y2)
            final_lines.append(("vertical", pt1, pt2, filtered))

            cv2.line(output, pt1, pt2, (255, 0, 0), 2)
            for x, y in filtered:
                cv2.circle(output, (x, y), 3, (0, 255, 0), -1)

    return output, final_lines


if __name__ == "__main__":
    result_img, lines = detect_dotted_lines_hv("image.png", debug=False)

    print(f"Detected {len(lines)} dotted line(s)")
    for idx, (direction, p1, p2, group) in enumerate(lines, 1):
        print(f"Line {idx}: {direction} {p1} -> {p2}, dots={len(group)}")

    # ✅ SAVE OUTPUT IMAGE
    output_path = "output_dotted.png"
    cv2.imwrite(output_path, result_img)
    print(f"Output image saved to {output_path}")