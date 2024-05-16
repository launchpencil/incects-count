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

    # Draw contours as ellipses and count them
    result_image = image.copy()
    insect_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area:
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(result_image, ellipse, (0, 255, 0), 2)
            insect_count += 1

    return result_image, insect_count

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
