import streamlit as st
import cv2
import numpy as np

def count_insects(image, min_contour_area=300):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

    # Denoising
    denoised_image = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)

    # Fill holes
    filled_image = fill_holes(denoised_image)

    st.image(filled_image, caption='処理後の画像', use_column_width=True)

    # Find contours
    contours, _ = cv2.findContours(denoised_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    # Draw contours
    result_image = image.copy()
    insect_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_image, f'{w}x{h}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            insect_count += 1

    return result_image, insect_count

def fill_holes(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_image = np.zeros_like(image)
    cv2.drawContours(filled_image, contours, -1, 255, -1)
    return filled_image

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
