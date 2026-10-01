import cv2
import numpy as np

def get_dominant_color(roi):
    """領域内の平均色を計算し、指定された6色を判定する"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    
    # 枠線の黒や影などのノイズを避けるため、領域の中心付近だけをサンプリング
    center_hsv = hsv[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    mean_hsv = np.mean(center_hsv, axis=(0, 1))
    
    hue, sat, val = mean_hsv[0], mean_hsv[1], mean_hsv[2]

    if sat < 60 and val > 120: return "White"
    if val < 50: return "Empty/Black" # 手の影などで暗い場合
        
    if hue < 10 or hue > 160: return "Red"
    elif 10 <= hue < 35: return "Yellow"
    elif 35 <= hue < 85: return "Green"
    elif 85 <= hue < 110: return "Light Blue"
    elif 110 <= hue < 140: return "Navy"
    else: return "Unknown"

def find_cross_box(frame):
    """画像からバツ印のある四角形を探して座標を返す"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 輪郭を見つける
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 20000: # 想定される枠の大きさ（環境に合わせて調整）
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # 四角形であるかチェック
            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                
                # 【重要】他の四角形と区別するため、枠の内側の「線の量(エッジ)」を調べる
                margin = int(w * 0.15) # 枠線自体を含めないように内側を切り抜く
                inner_roi = edges[y+margin : y+h-margin, x+margin : x+w-margin]
                
                if inner_roi.size > 0:
                    # 内側に白いピクセル（エッジ）がどれくらいあるか割合を計算
                    edge_density = np.count_nonzero(inner_roi) / inner_roi.size
                    
                    # バツ印があればエッジの密度が高くなる（0.05=5%以上をバツ枠とみなす）
                    if edge_density > 0.05:
                        return (x, y, w, h)
    return None

def main():
    cap = cv2.VideoCapture(0)
    last_known_box = None  # 最後に確認したバツ枠の座標
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 毎フレーム、バツ枠を探す
        current_box = find_cross_box(frame)
        
        if current_box is not None:
            # バツ枠が見えている間は、常に最新の座標に更新（記憶）し続ける
            last_known_box = current_box
            x, y, w, h = current_box
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Target Visible", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        else:
            # バツ枠が見つからない ＝ 手で隠された or さいころが置かれた状態
            if last_known_box is not None:
                # 最後に記憶した座標をロックして色判定を行う
                x, y, w, h = last_known_box
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) # ロック中は赤枠
                
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    color = get_dominant_color(roi)
                    cv2.putText(frame, f"Locked. Color: {color}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Searching for Target...", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Camera", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()