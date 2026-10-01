import cv2
import numpy as np

def get_dominant_color(roi):
    """領域内の平均色を計算し、指定された6色+αを判定する"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    
    # 枠線の黒や影などのノイズを避けるため、領域の中心付近だけをサンプリング
    center_hsv = hsv[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    mean_hsv = np.mean(center_hsv, axis=(0, 1))
    
    hue, sat, val = mean_hsv[0], mean_hsv[1], mean_hsv[2]

    # 1. 彩度(Saturation)と明度(Value)による「白」の判定
    # 彩度が低く、かつ明るい場合は「白」
    if sat < 60 and val > 120:
        return "White"
        
    # （※さいころが置かれていない時の「黒いバツ印」や影を誤検知しないための保険）
    if val < 50:
        return "Empty/Black"
        
    # 2. 色相(Hue)による有彩色の判定 (OpenCVのHueは 0〜179)
    if hue < 10 or hue > 160:
        return "Red"
    elif 10 <= hue < 35:
        return "Yellow"
    elif 35 <= hue < 85:
        return "Green"
    elif 85 <= hue < 110:
        # 水色は緑(約60)と青(約120)の中間付近
        return "Light Blue"
    elif 110 <= hue < 140:
        # 紺色（濃い青）は青の中心付近
        return "Navy"
    else:
        return "Unknown"

def main():
    cap = cv2.VideoCapture(0)
    target_roi = None  # 枠の座標を保存する変数 (x, y, w, h)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if target_roi is None:
            # 【初期化フェーズ】枠の場所を記憶する
            cv2.putText(frame, "Press 's' to select the target box, or 'q' to quit.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Camera", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # 's'キーを押すと画面が止まるので、マウスでバツ枠をドラッグして囲み、Enterキーで決定
                target_roi = cv2.selectROI("Camera", frame, showCrosshair=True, fromCenter=False)
                # もし自動検出にしたい場合は、ここで輪郭検出を行い target_roi = (x, y, w, h) を代入します
                
        else:
            # 【実行フェーズ】記憶した枠の位置だけを監視・色判定する
            x, y, w, h = target_roi
            
            # 監視している枠を画面に描画
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 記憶した領域（ROI）だけを切り出して色判定
            roi = frame[y:y+h, x:x+w]
            
            if roi.size > 0:
                color = get_dominant_color(roi)
                cv2.putText(frame, f"Dice: {color}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            cv2.imshow("Camera", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()