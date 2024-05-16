import streamlit as st
import cv2
import numpy as np

def count_insects(image, min_contour_area=200):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    # Contour detection using ellipses
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours as ellipses, display size, and merge overlapping ellipses
    result_image = image.copy()
    insect_count = 0

    bounding_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area:
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(result_image, ellipse, (0, 255, 0), 2)

            # Display size of ellipse
            (x, y), (MA, ma), angle = ellipse
            size_text = f'{int(MA)}x{int(ma)}'
            cv2.putText(result_image, size_text, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            bounding_boxes.append(ellipse)

    # Merge overlapping ellipses
    merged_boxes = merge_ellipses(bounding_boxes)

    for box in merged_boxes:
        cv2.ellipse(result_image, box, (0, 255, 0), 2)
        insect_count += 1

    return result_image, insect_count

def merge_ellipses(ellipses, overlap_threshold=0.2):
    merged_ellipses = []
    for ellipse in ellipses:
        found_overlap = False
        for idx, merged_ellipse in enumerate(merged_ellipses):
            if overlap_ratio(ellipse, merged_ellipse, overlap_threshold):  # Pass overlap_threshold as an argument
                merged_ellipses[idx] = merge_ellipse(ellipse, merged_ellipse)
                found_overlap = True
                break
        if not found_overlap:
            merged_ellipses.append(ellipse)
    return merged_ellipses

def overlap_ratio(ellipse1, ellipse2, overlap_threshold):  # Add overlap_threshold as an argument
    intersect_area = np.pi * ellipse1[1][0] * ellipse1[1][1] * overlap_ratio(ellipse2[1][0] * ellipse2[1][1])
    area1 = np.pi * ellipse1[1][0] * ellipse1[1][1]
    area2 = np.pi * ellipse2[1][0] * ellipse2[1][1]
    return intersect_area / min(area1, area2)

def merge_ellipse(ellipse1, ellipse2):
    new_center = ((ellipse1[0][0] + ellipse2[0][0]) / 2, (ellipse1[0][1] + ellipse2[0][1]) / 2)
    new_axes = ((ellipse1[1][0] + ellipse2[1][0]) / 2, (ellipse1[1][1] + ellipse2[1][1]) / 2)
    new_angle = (ellipse1[2] + ellipse2[2]) / 2
    return new_center, new_axes, new_angle

def main():
    st.title("昆虫カウンター")
    
    uploaded_file = st.file_uploader("画像ファイルをアップロードしてください", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        st.image(image, caption='アップロードされた画像', use_column_width=True)
        
        result_image, insect_count = count_insects(image)
        st.image(result_image, caption='結果画像', use_column_width=True)
        st.write(f"昆虫の数: {insect_count}")

if __name__ == "__main__":
    main()
