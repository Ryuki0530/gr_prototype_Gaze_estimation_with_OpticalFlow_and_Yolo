import cv2
import numpy as np
import matplotlib.pyplot as plt

def estimate_camera_movement(prev_frame, curr_frame):
    """
    2枚の連続フレームから Farneback Optical Flow を計算し、
    カメラの x方向, y方向の移動量を推定（ピクセル単位）して返す。
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Farneback 法によるオプティカルフローを計算
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        0.5,   # ピラミッドスケール
        3,     # ピラミッドレイヤー数
        15,    # 窓サイズ
        3,     # イテレーション回数
        5,     # 多項式近似の拡大サイズ
        1.2,   # 多項式近似の定数
        0      # オプションフラグ
    )

    # flow[..., 0]: x方向, flow[..., 1]: y方向 のフロー
    dx = np.mean(flow[..., 0])
    dy = np.mean(flow[..., 1])

    return dx, dy


def run_camera_flow(plot_graph=True):
    """
    カメラから映像を取得しながら光フローを用いて (dx, dy) を推定。
    plot_graph = True の場合、matplotlib を用いて移動量の推移を
    リアルタイムにグラフ描画する。

    Parameters
    ----------
    plot_graph : bool, optional
        True であれば、(dx, dy) の推移をリアルタイムグラフ表示する。
        False であればグラフは表示しない。
    """
    
    # カメラ初期化
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが開けませんでした。")
        return

    # 最初のフレームを取得
    ret, prev_frame = cap.read()
    if not ret:
        print("最初のフレーム取得に失敗しました。")
        cap.release()
        return

    # グラフ表示用の初期設定
    if plot_graph:
        plt.ion()  # インタラクティブモード ON
        fig, ax = plt.subplots()
        ax.set_title("Camera movement (dx, dy)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Movement (pixels)")

        # プロット用リスト
        x_data = []
        dx_data = []
        dy_data = []

        # 2本のラインオブジェクトを用意
        line_dx, = ax.plot([], [], label='dx')
        line_dy, = ax.plot([], [], label='dy')
        ax.legend()

    frame_count = 0

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        # 光フローによる移動量推定
        dx, dy = estimate_camera_movement(prev_frame, curr_frame)

        # カメラ画像に移動量を描画
        cv2.putText(curr_frame,
                    f"dx: {dx:.2f}, dy: {dy:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2)
        cv2.imshow("OpticalFlow", curr_frame)

        # グラフ描画が有効なら更新
        if plot_graph:
            x_data.append(frame_count)
            dx_data.append(dx)
            dy_data.append(dy)

            # ラインデータを更新
            line_dx.set_xdata(x_data)
            line_dx.set_ydata(dx_data)
            line_dy.set_xdata(x_data)
            line_dy.set_ydata(dy_data)

            # スケールを自動調整
            ax.relim()
            ax.autoscale_view()

            # グラフ描画を更新
            plt.draw()
            plt.pause(0.01)

        frame_count += 1
        prev_frame = curr_frame.copy()

        # 'q' で終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # グラフ終了処理
    if plot_graph:
        plt.ioff()  # インタラクティブモード OFF
        plt.show()   # 最後にウィンドウを確定表示（閉じたくない場合）


if __name__ == "__main__":
    # グラフ描画を ON にして実行する場合
    run_camera_flow(plot_graph=True)

    # グラフ描画を OFF にして実行したい場合
    # run_camera_flow(plot_graph=False)
