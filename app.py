import streamlit as st
import cv2
import numpy as np

def count_insects(image, min_contour_area=200, max_aspect_ratio=3.0, min_aspect_ratio=0.3):
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 閾値処理を追加
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    
    # 輪郭を抽出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 輪郭を描画
    result_image = image.copy()
    insect_count = 0
    
    # 大きな楕円の輪郭内に含まれる小さな楕円を除外するためのリスト
    excluded_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area:
            # 楕円を近似して描画
            ellipse = cv2.fitEllipse(contour)
            aspect_ratio = ellipse[1][0] / ellipse[1][1]  # アスペクト比を計算
            if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                # 輪郭の中心点を取得
                center_x, center_y = ellipse[0]
                # 既存の除外リストに含まれるかチェック
                excluded = False
                for exc_contour in excluded_contours:
                    exc_ellipse = cv2.fitEllipse(exc_contour)
                    exc_center_x, exc_center_y = exc_ellipse[0]
                    distance = np.sqrt((center_x - exc_center_x)**2 + (center_y - exc_center_y)**2)
                    if distance < max(exc_ellipse[1][0], exc_ellipse[1][1]) / 2:
                        excluded = True
                        break
                if not excluded:
                    cv2.ellipse(result_image, ellipse, (0, 255, 0), 2)
                    insect_count += 1
                    # 大きな楕円の輪郭内に含まれる小さな楕円を除外するために追加
                    excluded_contours.append(contour)
    
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
