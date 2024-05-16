import streamlit as st
import cv2
import numpy as np

def count_insects(image, min_contour_area=200):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Thresholding
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    
    # Draw contours
    result_image = image.copy()
    insect_count = 0
    
    # Calculate ellipses and exclude smaller ones
    ellipses = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area:
            ellipse = cv2.fitEllipse(contour)
            ellipses.append(ellipse)
    
    # Merge overlapping ellipses
    merged_ellipses = merge_ellipses(ellipses)
    
    # Draw merged ellipses
    for ellipse in merged_ellipses:
        cv2.ellipse(result_image, ellipse, (0, 255, 0), 2)
        insect_count += 1
    
    return result_image, insect_count

def merge_ellipses(ellipses, overlap_threshold=0.2):
    # Merge overlapping ellipses
    merged_ellipses = []
    for ellipse in ellipses:
        found_overlap = False
        for idx, merged_ellipse in enumerate(merged_ellipses):
            # Calculate overlap ratio based on areas
            overlap_area = cv2.ellipseOverlap(ellipse, merged_ellipse)
            if overlap_area > overlap_threshold:
                # Merge ellipses if overlap is significant
                merged_ellipse = cv2.minAreaRect(np.concatenate((cv2.ellipse2Poly(ellipse), cv2.ellipse2Poly(merged_ellipse))))
                merged_ellipses[idx] = merged_ellipse
                found_overlap = True
                break
        if not found_overlap:
            merged_ellipses.append(ellipse)
    return merged_ellipses

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
