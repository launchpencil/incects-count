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
        # 斜めに描画するための回転角度を指定
        angle_deg = 30  # 任意の角度を設定
        angle_rad = np.radians(angle_deg)
        center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        size = (box[2] - box[0], box[3] - box[1])
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1)
        rotated_box = cv2.transform(np.array([box]), rotation_matrix)[0][0]

        # 長方形のサイズを少し小さくする
        margin = 1  # 任意のマージンを設定
        rotated_box[0] += margin
        rotated_box[1] += margin
        rotated_box[2] -= margin
        rotated_box[3] -= margin
        cv2.rectangle(result_image, (rotated_box[0], rotated_box[1]), (rotated_box[2], rotated_box[3]), (0, 255, 0), 2)
        cv2.putText(result_image, f'{rotated_box[2] - rotated_box[0]}x{rotated_box[3] - rotated_box[1]}', (rotated_box[0], rotated_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        insect_count += 1

    return result_image, insect_count

# 他の関数とmain()関数は省略

if __name__ == "__main__":
    main()
