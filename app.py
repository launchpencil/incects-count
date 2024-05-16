import streamlit as st
import cv2
import numpy as np

def count_insects(image, min_contour_area=200):
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 閾値処理を追加
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    # 輪郭を抽出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    # 輪郭を描画
    result_image = image.copy()
    insect_count = 0

    # 境界ボックスを計算し、重複するボックスを結合
    bounding_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area:
            # 楕円を近似
            ellipse = cv2.fitEllipse(contour)
            bounding_boxes.append(ellipse)

    # 重複するボックスを結合
    merged_boxes = merge_boxes(bounding_boxes)

    # 結合されたボックスを描画
    for box in merged_boxes:
        cv2.ellipse(result_image, box, (0, 255, 0), 2)
        insect_count += 1

    return result_image, insect_count

def merge_boxes(boxes, overlap_threshold=0.2):
    # ボックスの結合
    merged_boxes = []
    for box in boxes:
        found_overlap = False
        for idx, merged_box in enumerate(merged_boxes):
            # 重複する場合は結合する
            if overlap_ratio(box, merged_box) > overlap_threshold:
                merged_boxes[idx] = merge_ellipses(box, merged_box)
                found_overlap = True
                break
        if not found_overlap:
            merged_boxes.append(box)
    return merged_boxes

def overlap_ratio(box1, box2):
    # 重複する面積の割合を計算
    area1 = np.pi * box1[1][0] * box1[1][1]
    area2 = np.pi * box2[1][0] * box2[1][1]
    intersect_area = cv2.rotatedRectangleIntersection(box1, box2)[1]
    return intersect_area / min(area1, area2)

def merge_ellipses(box1, box2):
    # 楕円を結合する
    center1, axes1, angle1 = box1
    center2, axes2, angle2 = box2
    new_center = ((center1[0] + center2[0]) / 2, (center1[1] + center2[1]) / 2)
    new_axes = ((axes1[0] + axes2[0]) / 2, (axes1[1] + axes2[1]) / 2)
    new_angle = (angle1 + angle2) / 2
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
