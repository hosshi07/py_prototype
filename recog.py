import cv2
import numpy as np

def detect_squares(image, min_area=500):
    """画像から四角形の輪郭を検出する関数"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # エッジ検出
    edges = cv2.Canny(blurred, 50, 150)
    
    # 輪郭の抽出
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    squares = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            # 輪郭の近似（頂点数を減らす）
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # 頂点が4つ（四角形）で凸包であるか確認
            if len(approx) == 4 and cv2.isContourConvex(approx):
                squares.append((area, approx))
                
    # 面積が大きい順にソートして返す
    squares = sorted(squares, key=lambda x: x[0], reverse=True)
    return [s[1] for s in squares]

def get_dominant_color(roi):
    """領域内の平均色を計算し、色名を返す関数"""
    # RGBからHSVに変換（照明の変化に強いため）
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 領域の中央部分の平均値をとる
    h, w = hsv.shape[:2]
    center_hsv = hsv[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    mean_hsv = np.mean(center_hsv, axis=(0, 1))
    
    hue = mean_hsv[0]
    sat = mean_hsv[1]
    val = mean_hsv[2]

    # 彩度や明度が低い場合は白黒判定
    if sat < 50 and val > 100:
        return "White"
    if val < 50:
        return "Black"
        
    # 色相(Hue)による判定 (OpenCVのHueは0-179)
    if hue < 10 or hue > 160:
        return "Red"
    elif 10 <= hue < 35:
        return "Yellow"
    elif 35 <= hue < 85:
        return "Green"
    elif 85 <= hue < 130:
        return "Blue"
    else:
        return "Unknown"

def main():
    cap = cv2.VideoCapture(0) # 0番のカメラを使用
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 画像から四角形を検出
        squares = detect_squares(frame, min_area=1000)
        
        if len(squares) >= 1:
            # 1番大きい四角形を「地面の枠」と仮定
            frame_box = squares[0]
            cv2.polylines(frame, [frame_box], True, (0, 255, 0), 2)
            cv2.putText(frame, "Frame", tuple(frame_box[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # もし2つ以上の四角形が見つかった場合、2番目を「さいころ」と仮定
            if len(squares) >= 2:
                dice_box = squares[1]
                
                # さいころが枠の中にあるかどうかの判定（中心座標で簡易的に判定）
                M = cv2.moments(dice_box)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # さいころの中心が枠のポリゴン内にあるかチェック
                    if cv2.pointPolygonTest(frame_box, (cX, cY), False) >= 0:
                        cv2.polylines(frame, [dice_box], True, (255, 0, 0), 2)
                        
                        # さいころのバウンディングボックスを取得して画像を切り抜く（ROI）
                        x, y, w, h = cv2.boundingRect(dice_box)
                        dice_roi = frame[y:y+h, x:x+w]
                        
                        if dice_roi.size > 0:
                            # 色の判定
                            color_name = get_dominant_color(dice_roi)
                            cv2.putText(frame, f"Dice: {color_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Dice Detection", frame)
        
        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()