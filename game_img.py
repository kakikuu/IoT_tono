import cv2
import mediapipe as mp
import pyfirmata
import time
import random

# Arduinoの設定
b = pyfirmata.Arduino("/dev/cu.usbmodem201912341")  # Arduinoボードの設定
it = pyfirmata.util.Iterator(b)
it.start()
led_pin = b.get_pin("d:2:o")  # デジタルピン2にLEDを接続

# 初期得点設定
score = 10  # 初期得点

# Mediapipeの設定
mp_drawing = mp.solutions.drawing_utils  # 描画ユーティリティ
mp_face_mesh = mp.solutions.face_mesh  # FaceMeshの設定

cap = cv2.VideoCapture(0)  # カメラキャプチャの開始
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)  # 顔検出の信頼度設定

# 障害物の画像を読み込む
obstacle_images = [
    cv2.imread("./job_senesi.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("./job_taiiku_man.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("./pose_galpeace_schoolgirl.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("./yokokara_shitsurei1_schoolboy.png", cv2.IMREAD_UNCHANGED),
]

# 画像の読み込み状態を確認
for i, img in enumerate(obstacle_images):
    if img is None:
        print(f"Image {i} could not be loaded.")
    else:
        print(f"Image {i} loaded with shape: {img.shape}")

# 障害物の設定
num_obstacles = 8
obstacles = []
for _ in range(num_obstacles):
    x = random.randint(0, 640)  # 初期x座標
    y = random.randint(-480, 0)  # 初期y座標（画面外からスタート）
    width = random.randint(50, 100)  # 幅
    height = random.randint(50, 100)  # 高さ
    speed_x = random.choice([-1, 1]) * random.uniform(1.5, 20.0)  # 斜め移動用
    speed_y = random.uniform(1.0, 70.0)  # 下方向の移動速度
    image_index = random.randint(
        0, len(obstacle_images) - 1
    )  # 使用する画像をランダムに選択
    obstacles.append([x, y, width, height, speed_x, speed_y, image_index])


# 衝突検出関数
def check_collision(person_coords, obstacle_coords):
    px, py = person_coords
    ox, oy, ow, oh = obstacle_coords[:4]
    return (ox <= px <= ox + ow) and (oy <= py <= oy + oh)


# カウントダウンを表示する関数
def display_countdown(seconds):
    for i in range(seconds, 0, -1):
        print(f"Game starts in {i}...")
        time.sleep(1)


# カウントダウンの表示
display_countdown(3)

# メインループ
while cap.isOpened():
    success, image = cap.read()  # カメラからのフレームを読み込む
    if not success:
        print("カメラの読み込みに失敗しました。")
        continue

    image = cv2.cvtColor(
        cv2.flip(image, 1), cv2.COLOR_BGR2RGB
    )  # 画像をRGBに変換して反転
    results = face_mesh.process(image)  # 顔の特徴点を検出

    if results.multi_face_landmarks:
        h, w, _ = image.shape
        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                lm_x = int(lm.x * w)
                lm_y = int(lm.y * h)

                # 障害物との衝突検出
                for obstacle in obstacles:
                    if check_collision((lm_x, lm_y), obstacle):
                        print("衝突しました！得点が減少します。")
                        score -= 2  # 衝突した場合の減点
                        led_pin.write(1)  # LEDを点灯
                        time.sleep(1)
                        led_pin.write(0)  # LEDを消灯

                        # 衝突した障害物のみリセット
                        obstacle[0] = random.randint(0, 640)  # 新しいx座標
                        obstacle[1] = random.randint(-480, 0)  # 新しいy座標
                        obstacle[4] = random.choice([-1, 1]) * random.uniform(
                            1.5, 20.0
                        )  # 新しい斜め移動用速度
                        obstacle[5] = random.uniform(
                            1.0, 70.0
                        )  # 新しい下方向の移動速度
                        obstacle[6] = random.randint(
                            0, len(obstacle_images) - 1
                        )  # 新しい画像の選択

                        if score <= 0:
                            print("ゲームオーバー！最終得点:", score)
                            cap.release()
                            cv2.destroyAllWindows()
                            exit()

    # 障害物の移動と再描画
    for obstacle in obstacles:
        obstacle[0] += obstacle[4]  # x座標の移動
        obstacle[1] += obstacle[5]  # y座標の移動
        if obstacle[1] > 480:  # 画面下に出たらリセット
            obstacle[0] = random.randint(0, 640)
            obstacle[1] = random.randint(-480, 0)
            obstacle[4] = random.choice([-1, 1]) * random.uniform(0.5, 3.0)
            obstacle[5] = random.uniform(1.0, 5.0)
            obstacle[6] = random.randint(
                0, len(obstacle_images) - 1
            )  # 新しい画像の選択

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    # 障害物の描画
    for obstacle in obstacles:
        x, y, w, h = map(int, obstacle[:4])
        image_index = obstacle[6]
        resized_obstacle = cv2.resize(obstacle_images[image_index], (w, h))

        if resized_obstacle.shape[0] > 0 and resized_obstacle.shape[1] > 0:
            if resized_obstacle.shape[2] == 4:  # アルファチャンネルが存在する場合
                alpha_s = resized_obstacle[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                # 画像の範囲をチェック
                x = max(0, x)
                y = max(0, y)
                if y + h > image.shape[0] or x + w > image.shape[1]:
                    print(
                        f"Skipping drawing image {image_index} as it exceeds boundaries."
                    )
                    continue

                h, w = alpha_s.shape[:2]

                print(f"Drawing image {image_index} at ({x}, {y}) with size ({w}, {h})")

                for c in range(0, 3):
                    try:
                        image[y : y + h, x : x + w, c] = (
                            alpha_s[:h, :w] * resized_obstacle[:h, :w, c]
                            + alpha_l[:h, :w] * image[y : y + h, x : x + w, c]
                        )
                    except Exception as e:
                        print(f"Error while blending image: {e}")
            else:
                print(
                    f"Drawing image {image_index} without alpha channel at ({x}, {y}) with size ({w}, {h})"
                )
                if resized_obstacle.shape[0] > 0 and resized_obstacle.shape[1] > 0:
                    try:
                        image[y : y + h, x : x + w] = resized_obstacle[:h, :w]
                    except Exception as e:
                        print(f"Error while placing image: {e}")

    cv2.imshow("MediaPipe FaceMesh", image)  # 画像を表示

    if cv2.waitKey(1) == ord("q"):  # 'q'キーが押されたら終了
        break

cap.release()
cv2.destroyAllWindows()
