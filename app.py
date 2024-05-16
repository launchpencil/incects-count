import streamlit as st
import cv2
import numpy as np
from io import BytesIO

def count_insects(image):
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 画像の平滑化（ガウシアンブラー）
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 適応的閾値処理（パラメータ調整）
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 7)

    st.image(binary, caption='閾値処理された画像', use_column_width=True)

    # 輪郭の抽出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭の周囲に矩形を描画し、小さな領域を除去
    result_image = image.copy()
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # 例：面積が50ピクセルより大きい輪郭を保持
            valid_contours.append(contour)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 昆虫の数を数える
    insect_count = len(valid_contours)

    return result_image, insect_count


def main():
    st.title("昆虫数を数えるアプリ")
    st.write("画像から昆虫の数を数える")
    
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