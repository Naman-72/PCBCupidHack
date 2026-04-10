import cv2
import numpy as np


def resize_for_display(image, max_width=1200):
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (int(w * scale), int(h * scale)))


def find_breadboard_mask(image, debug=False):
    """
    Detect the breadboard body using brightness and low saturation.
    Breadboards are usually light-colored and less saturated than wires/components.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Light / low-saturation region
    lower = np.array([0, 0, 80], dtype=np.uint8)
    upper = np.array([180, 80, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    breadboard_mask = np.zeros_like(mask)
    breadboard_mask[labels == largest_idx] = 255

    # Fill holes to get solid board region
    contours, _ = cv2.findContours(breadboard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    if debug:
        print("Breadboard mask detected")

    return filled


def detect_foreign_objects(image, breadboard_mask, debug=False):
    """
    Detect components/wires on top of breadboard.
    We combine:
    - high saturation objects (colored wires)
    - dark objects (ICs, shadows, components)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Colored objects
    sat_mask = cv2.inRange(hsv[:, :, 1], 60, 255)

    # Dark objects
    dark_mask = cv2.inRange(gray, 0, 90)

    # Combine
    obj_mask = cv2.bitwise_or(sat_mask, dark_mask)

    # Restrict to breadboard region only
    obj_mask = cv2.bitwise_and(obj_mask, breadboard_mask)

    # Remove tiny breadboard holes so they are not treated as objects
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # Open removes tiny dots
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Close expands component regions a bit
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel_big, iterations=2)

    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)
    cleaned = np.zeros_like(obj_mask)

    min_area = 80
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    if debug:
        print("Foreign object mask detected")

    return cleaned


def inpaint_objects(image, object_mask):
    """
    Remove instruments/components using inpainting.
    """
    cleaned = cv2.inpaint(image, object_mask, 5, cv2.INPAINT_TELEA)
    return cleaned


def crop_to_mask(image, mask):
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]


def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    breadboard_mask = find_breadboard_mask(image, debug=True)
    object_mask = detect_foreign_objects(image, breadboard_mask, debug=True)
    cleaned = inpaint_objects(image, object_mask)

    # Optional: crop only board region
    board_only = cv2.bitwise_and(image, image, mask=breadboard_mask)
    cleaned_board_only = cv2.bitwise_and(cleaned, cleaned, mask=breadboard_mask)

    cv2.imwrite("01_breadboard_mask.png", breadboard_mask)
    cv2.imwrite("02_object_mask.png", object_mask)
    cv2.imwrite("03_cleaned_inpainted.png", cleaned)
    cv2.imwrite("04_board_only.png", board_only)
    cv2.imwrite("05_cleaned_board_only.png", cleaned_board_only)

    print("Saved:")
    print("  01_breadboard_mask.png")
    print("  02_object_mask.png")
    print("  03_cleaned_inpainted.png")
    print("  04_board_only.png")
    print("  05_cleaned_board_only.png")


if __name__ == "__main__":
    main("breadboard.jpg")