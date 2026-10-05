import cv2
import numpy as np
import math

# 左から順に並んでいる前提の色名リスト（画像に合わせて「黄, 赤, 緑, 水色, 白, 紺色」に設定）
COLOR_NAMES = ["Yellow", "Red", "Green", "Light Blue", "White", "Navy"]

def get_lab_color(roi):
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    center_lab = lab[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    return np.mean(center_lab, axis=(0, 1))

def color_distance(color1, color2):
    return np.linalg.norm(color1 - color2)

def get_closest_color_name(roi, ref_colors):
    if not ref_colors:
        return "Uncalibrated"
    current_color = get_lab_color(roi)
    distances = [color_distance(current_color, ref) for ref in ref_colors]
    min_index = np.argmin(distances)
    return COLOR_NAMES[min_index]

def detect_field_elements(frame):
    """画像内の全四角形を検出し、バツ枠とカラー枠に振り分ける"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    squares = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                cx, cy = x + w//2, y + h//2
                
                # 重複チェック（近い四角形は除外）
                is_duplicate = False
                for sq in squares:
                    sq_cx = sq['rect'][0] + sq['rect'][2]//2
                    sq_cy = sq['rect'][1] + sq['rect'][3]//2
                    if math.hypot(cx - sq_cx, cy - sq_cy) < 20:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    squares.append({'rect': (x, y, w, h), 'area': area})
                
    if not squares:
        return None, []

    cross_box = None
    color_boxes = []
    
    # 面積の大きい順にチェックしていく
    squares.sort(key=lambda s: s['area'], reverse=True)
    
    for s in squares:
        x, y, w, h = s['rect']
        margin = int(w * 0.15)
        
        # まだバツ枠が確定しておらず、かつ枠が十分な大きさの場合
        if cross_box is None and w > margin*2 and h > margin*2:
            inner_roi = edges[y+margin : y+h-margin, x+margin : x+w-margin]
            if inner_roi.size > 0:
                edge_density = np.count_nonzero(inner_roi) / inner_roi.size
                # 内側に5%以上のエッジ（線）があれば「バツ枠」として確定
                if edge_density > 0.08:
                    cross_box = s['rect']
                    continue
        
        # バツ枠以外の条件を満たさないものはすべて「カラー枠」
        color_boxes.append(s['rect'])

    return cross_box, color_boxes

def sort_color_boxes(color_boxes):
    """カメラが固定されているため、単純にX座標（左から右）でソートする"""
    return sorted(color_boxes, key=lambda b: b[0])

def main():
    cap = cv2.VideoCapture(0)
    
    last_known_cross = None
    reference_colors = []
    is_calibrated = False

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 毎フレーム、バツ枠とカラー枠を探す
        cross_box, color_boxes = detect_field_elements(frame)
        
        if not is_calibrated:
            cv2.putText(frame, "Press 'c' to Calibrate Colors", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # キャリブレーション待ちの描画
            if cross_box:
                cx, cy, cw, ch = cross_box
                cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 2)
            for box in color_boxes:
                sx, sy, sw, sh = box
                cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (255, 255, 0), 2)
                
            # 'c'キーでキャリブレーション実行
            if cv2.waitKey(1) & 0xFF == ord('c'):
                if cross_box and len(color_boxes) >= 6:
                    sorted_boxes = sort_color_boxes(color_boxes)
                    reference_colors = [get_lab_color(frame[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]) for b in sorted_boxes[:6]]
                    
                    is_calibrated = True
                    print("Calibration Successful!")
                else:
                    print(f"Calibration Failed: Found {len(color_boxes)} color boxes.")
        
        else:
            if cross_box is not None:
                cx, cy, cw, ch = cross_box
                cv2.rectangle(frame, (cx-10, cy-10), (cx+cw+10, cy+ch+10), (0, 0, 255), 2)
                roi = frame[cy:cy+ch, cx:cx+cw]
                if roi.size > 0:
                    color_name = get_closest_color_name(roi, reference_colors)
                    cv2.putText(frame, f"Color: {color_name}", (cx, cy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
        cv2.imshow("Field Camera", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            is_calibrated = False
            reference_colors = []

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()