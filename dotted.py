import cv2
import numpy as np
from math import atan2, degrees


def detect_dotted_lines(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary inverse: dark dots become white blobs
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Remove tiny noise and slightly clean blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours = candidate dots
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5 or area > 200:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Dot-like filtering
        aspect_ratio = w / float(h)
        if 0.5 < aspect_ratio < 2.0:
            cx = x + w // 2
            cy = y + h // 2
            centers.append((cx, cy))

            if debug:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 1)

    if len(centers) < 2:
        return output, []

    points = np.array(centers, dtype=np.int32)

    # Group points into candidate dotted lines
    used = set()
    lines_found = []

    max_angle_diff = 10      # degrees
    max_dist_to_line = 10    # pixels
    min_points_in_line = 4

    for i in range(len(points)):
        if i in used:
            continue

        best_group = []

        for j in range(i + 1, len(points)):
            p1 = points[i]
            p2 = points[j]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 0 and dy == 0:
                continue

            base_angle = degrees(atan2(dy, dx))

            group = [tuple(p1), tuple(p2)]

            for k in range(len(points)):
                if k == i or k == j:
                    continue

                p = points[k]

                # Distance from point to line through p1-p2
                num = abs((p2[1] - p1[1]) * p[0] - (p2[0] - p1[0]) * p[1] + p2[0]*p1[1] - p2[1]*p1[0])
                den = np.hypot(p2[1] - p1[1], p2[0] - p1[0])
                dist = num / den if den != 0 else 999

                # Angle consistency
                angle = degrees(atan2(p[1] - p1[1], p[0] - p1[0]))
                angle_diff = abs(angle - base_angle)
                angle_diff = min(angle_diff, 180 - angle_diff)

                if dist < max_dist_to_line and angle_diff < max_angle_diff:
                    group.append(tuple(p))

            # Keep only meaningful groups
            group = list(set(group))
            if len(group) > len(best_group):
                best_group = group

        if len(best_group) >= min_points_in_line:
            # Mark used points
            for pt in best_group:
                idx = np.where((points == pt).all(axis=1))[0]
                for idv in idx:
                    used.add(int(idv))

            lines_found.append(best_group)

    # Draw fitted lines
    final_lines = []
    for group in lines_found:
        group_np = np.array(group, dtype=np.float32)
        [vx, vy, x0, y0] = cv2.fitLine(group_np, cv2.DIST_L2, 0, 0.01, 0.01)

        vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

        # Project points onto fitted direction to get endpoints
        projections = []
        for x, y in group:
            t = (x - x0) * vx + (y - y0) * vy
            projections.append((t, x, y))

        projections.sort()
        t_min = projections[0][0]
        t_max = projections[-1][0]

        pt1 = (int(x0 + t_min * vx), int(y0 + t_min * vy))
        pt2 = (int(x0 + t_max * vx), int(y0 + t_max * vy))

        final_lines.append((pt1, pt2, group))
        cv2.line(output, pt1, pt2, (0, 0, 255), 2)

        for x, y in group:
            cv2.circle(output, (x, y), 3, (0, 255, 0), -1)

    return output, final_lines


if __name__ == "__main__":
    result_img, lines = detect_dotted_lines("image.png", debug=False)

    print(f"Detected {len(lines)} dotted line(s)")
    for idx, (p1, p2, group) in enumerate(lines, 1):
        print(f"Line {idx}: {p1} -> {p2}, dots={len(group)}")

    cv2.imshow("Detected Dotted Lines", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()