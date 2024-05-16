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
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, x + w, y + h))
    
    # 重複するボックスを結合
    merged_boxes = merge_boxes(bounding_boxes)
    
    # 結合されたボックスを描画
    for box in merged_boxes:
        x1, y1, x2, y2 = box
        # 長方形のサイズを少し小さくする
        margin = 10  # 任意のマージンを設定
        x1 += margin
        y1 += margin
        x2 -= margin
        y2 -= margin
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_image, f'{x2 - x1}x{y2 - y1}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        insect_count += 1

    return result_image, insect_count

def merge_boxes(boxes, overlap_threshold=0.2):
    # ボックスの結合
    merged_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        found_overlap = False
        # ボックスを縮小
        margin = 5  # 任意のマージンを設定
        x1 -= margin
        y1 -= margin
        x2 += margin
        y2 += margin
        for idx, merged_box in enumerate(merged_boxes):
            x1_m, y1_m, x2_m, y2_m = merged_box
            # 重複する場合は結合する
            if overlap_ratio((x1, y1, x2, y2), (x1_m, y1_m, x2_m, y2_m)) > overlap_threshold:
                merged_boxes[idx] = (
                    min(x1, x1_m),
                    min(y1, y1_m),
                    max(x2, x2_m),
                    max(y2, y2_m)
                )
                found_overlap = True
                break
        if not found_overlap:
            merged_boxes.append((x1, y1, x2, y2))
    return merged_boxes

def overlap_ratio(box1, box2):
    # 重複する面積の割合を計算
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    intersect_area = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    return intersect_area / min(area1, area2)

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