import cv2
import numpy as np


def cluster_1d(points, axis=0, tolerance=10):
    """
    axis=0 => cluster by x (vertical families)
    axis=1 => cluster by y (horizontal families)
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


def split_into_runs(sorted_group, axis, max_gap):
    """
    Split one clustered row/column into continuous runs.
    axis=0 => use x progression
    axis=1 => use y progression
    """
    if not sorted_group:
        return []

    runs = [[sorted_group[0]]]

    for i in range(1, len(sorted_group)):
        prev = sorted_group[i - 1]
        cur = sorted_group[i]

        if abs(cur[axis] - prev[axis]) <= max_gap:
            runs[-1].append(cur)
        else:
            runs.append([cur])

    return runs


def merge_intervals(intervals, merge_gap=10):
    """
    Merge overlapping or nearly-touching 1D intervals.
    intervals = [(start, end), ...]
    """
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda t: t[0])
    merged = [list(intervals[0])]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]

        if start <= last_end + merge_gap:
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])

    return [tuple(x) for x in merged]


def merge_lines_by_coordinate(lines, coord_index, start_index, end_index, coord_tolerance=8, merge_gap=10):
    """
    Generic merger for horizontal/vertical lines.

    line format:
      horizontal: (y, x1, x2, points)
      vertical:   (x, y1, y2, points)

    coord_index = 0  -> fixed coordinate (y for horizontal, x for vertical)
    start_index = 1
    end_index   = 2
    """
    if not lines:
        return []

    # Sort by fixed coordinate first
    lines = sorted(lines, key=lambda l: l[coord_index])

    grouped = [[lines[0]]]
    for line in lines[1:]:
        if abs(line[coord_index] - grouped[-1][-1][coord_index]) <= coord_tolerance:
            grouped[-1].append(line)
        else:
            grouped.append([line])

    merged_output = []

    for group in grouped:
        coord_vals = [g[coord_index] for g in group]
        fixed_coord = int(round(np.mean(coord_vals)))

        intervals = [(g[start_index], g[end_index]) for g in group]
        merged_intervals = merge_intervals(intervals, merge_gap=merge_gap)

        all_pts = []
        for g in group:
            all_pts.extend(g[3])

        for start, end in merged_intervals:
            # keep only points belonging to this merged interval
            pts = []
            for p in all_pts:
                axis_val = p[0] if coord_index == 0 else p[1]
                if start <= axis_val <= end:
                    pts.append(p)

            merged_output.append((fixed_coord, start, end, pts))

    return merged_output


def detect_dotted_lines_hv_merged(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Dark holes/dots become white
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5 or area > 200:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        if 0.5 < aspect_ratio < 2.0:
            cx = x + w // 2
            cy = y + h // 2
            centers.append((cx, cy))

            if debug:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 1)

    if len(centers) < 2:
        return output, [], []

    min_points_in_line = 4
    coord_tolerance = 10
    max_gap = 40
    merge_gap = 12

    horizontal_candidates = []
    vertical_candidates = []

    # -------------------------
    # Horizontal detection
    # -------------------------
    h_groups = cluster_1d(centers, axis=1, tolerance=coord_tolerance)

    for group in h_groups:
        if len(group) < min_points_in_line:
            continue

        group = sorted(group, key=lambda p: p[0])  # sort by x
        runs = split_into_runs(group, axis=0, max_gap=max_gap)

        for run in runs:
            if len(run) >= min_points_in_line:
                y = int(round(np.mean([p[1] for p in run])))
                x1 = min(p[0] for p in run)
                x2 = max(p[0] for p in run)
                horizontal_candidates.append((y, x1, x2, run))

    # -------------------------
    # Vertical detection
    # -------------------------
    v_groups = cluster_1d(centers, axis=0, tolerance=coord_tolerance)

    for group in v_groups:
        if len(group) < min_points_in_line:
            continue

        group = sorted(group, key=lambda p: p[1])  # sort by y
        runs = split_into_runs(group, axis=1, max_gap=max_gap)

        for run in runs:
            if len(run) >= min_points_in_line:
                x = int(round(np.mean([p[0] for p in run])))
                y1 = min(p[1] for p in run)
                y2 = max(p[1] for p in run)
                vertical_candidates.append((x, y1, y2, run))

    # -------------------------
    # Merge overlapping lines
    # -------------------------
    merged_horizontal = merge_lines_by_coordinate(
        horizontal_candidates,
        coord_index=0,   # y
        start_index=1,   # x1
        end_index=2,     # x2
        coord_tolerance=coord_tolerance,
        merge_gap=merge_gap
    )

    merged_vertical = merge_lines_by_coordinate(
        vertical_candidates,
        coord_index=0,   # x
        start_index=1,   # y1
        end_index=2,     # y2
        coord_tolerance=coord_tolerance,
        merge_gap=merge_gap
    )

    # -------------------------
    # Draw result
    # -------------------------
    for y, x1, x2, pts in merged_horizontal:
        cv2.line(output, (x1, y), (x2, y), (0, 0, 255), 2)
        for px, py in pts:
            cv2.circle(output, (px, py), 3, (0, 255, 0), -1)

    for x, y1, y2, pts in merged_vertical:
        cv2.line(output, (x, y1), (x, y2), (255, 0, 0), 2)
        for px, py in pts:
            cv2.circle(output, (px, py), 3, (0, 255, 0), -1)

    return output, merged_horizontal, merged_vertical


if __name__ == "__main__":
    input_path = "hungry.jpg"
    output_path = "output_merged.png"

    result_img, h_lines, v_lines = detect_dotted_lines_hv_merged(input_path, debug=False)

    print(f"Horizontal lines: {len(h_lines)}")
    for i, (y, x1, x2, pts) in enumerate(h_lines, 1):
        print(f"H{i}: y={y}, x1={x1}, x2={x2}, dots={len(pts)}")

    print(f"Vertical lines: {len(v_lines)}")
    for i, (x, y1, y2, pts) in enumerate(v_lines, 1):
        print(f"V{i}: x={x}, y1={y1}, y2={y2}, dots={len(pts)}")

    cv2.imwrite(output_path, result_img)
    print(f"Saved: {output_path}")