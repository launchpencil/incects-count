import streamlit as st
import cv2
import numpy as np
from io import BytesIO

def count_insects(image, min_contour_area=150):
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 閾値処理を追加
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    
    # 輪郭を抽出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    
    # 重なりを除外した輪郭を抽出
    cleaned_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area:
            cleaned_contours.append(contour)
    
    # 輪郭を描画
    result_image = image.copy()
    insect_count = 0
    for contour in cleaned_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 長方形の大きさを描画
        cv2.putText(result_image, f'{w}x{h}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        insect_count += 1
    
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
