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
    # 結合されたボックスを描画
    for box in merged_boxes:
        x, y, major_axis, minor_axis, angle = box
        rect = cv2.minAreaRect(np.array([(x, y), (major_axis, minor_axis), angle], dtype=np.float32))
        box_points = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(result_image, [box_points], 0, (0, 255, 0), 2)
        cv2.putText(result_image, f'{major_axis}x{minor_axis}', (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        insect_count += 1

    
    return result_image, insect_count

def merge_boxes(boxes, area_threshold=0.2):
    # Sort boxes based on area in descendQing order
    boxes = sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
    
    merged_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        merged = False
        for m_box in merged_boxes:
            mx1, my1, mx2, my2 = m_box
            # Calculate overlap ratio based on area
            overlap_area = max(0, min(x2, mx2) - max(x1, mx1)) * max(0, min(y2, my2) - max(y1, my1))
            box_area = (x2 - x1) * (y2 - y1)
            m_box_area = (mx2 - mx1) * (my2 - my1)
            if overlap_area / min(box_area, m_box_area) > area_threshold:
                merged_boxes.remove(m_box)
                merged_boxes.append((
                    min(x1, mx1),
                    min(y1, my1),
                    max(x2, mx2),
                    max(y2, my2)
                ))
                merged = True
                break
        if not merged:
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