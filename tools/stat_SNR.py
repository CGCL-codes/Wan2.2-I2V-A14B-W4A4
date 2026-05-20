import cv2
import csv
import argparse
import numpy as np
from pathlib import Path


def calculate_laplacian_variance(frame):
    """
    计算图像的 Laplacian 方差，用于衡量清晰度。

    数值越大，通常表示图像越清晰；
    数值越小，通常表示图像越模糊。
    """

    if frame is None:
        raise ValueError("输入帧为空")

    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    gray = gray.astype(np.float64)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    return laplacian.var()


def analyze_video_sharpness(clean_video_path, test_video_path, output_csv):
    clean_video_path = Path(clean_video_path)
    test_video_path = Path(test_video_path)

    if not clean_video_path.exists():
        raise FileNotFoundError(f"干净参考视频不存在: {clean_video_path}")

    if not test_video_path.exists():
        raise FileNotFoundError(f"测试视频不存在: {test_video_path}")

    clean_cap = cv2.VideoCapture(str(clean_video_path))
    test_cap = cv2.VideoCapture(str(test_video_path))

    if not clean_cap.isOpened():
        raise RuntimeError(f"无法打开干净参考视频: {clean_video_path}")

    if not test_cap.isOpened():
        raise RuntimeError(f"无法打开测试视频: {test_video_path}")

    clean_fps = clean_cap.get(cv2.CAP_PROP_FPS)
    test_fps = test_cap.get(cv2.CAP_PROP_FPS)

    clean_total_frames = int(clean_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    test_total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clean_width = int(clean_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    clean_height = int(clean_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    test_width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    test_height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"干净视频: {clean_width}x{clean_height}, FPS={clean_fps}, 帧数={clean_total_frames}")
    print(f"测试视频: {test_width}x{test_height}, FPS={test_fps}, 帧数={test_total_frames}")

    if clean_width != test_width or clean_height != test_height:
        print("警告: 两个视频分辨率不同，将把测试视频 resize 到干净视频尺寸后再比较。")

    if abs(clean_fps - test_fps) > 1e-3:
        print("警告: 两个视频 FPS 不一致，当前脚本仍按帧序号逐帧比较。")

    total_frames = min(clean_total_frames, test_total_frames)

    results = []

    frame_index = 0

    while True:
        clean_ret, clean_frame = clean_cap.read()
        test_ret, test_frame = test_cap.read()

        if not clean_ret or not test_ret:
            break

        if clean_frame.shape != test_frame.shape:
            test_frame = cv2.resize(
                test_frame,
                (clean_frame.shape[1], clean_frame.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        clean_lap = calculate_laplacian_variance(clean_frame)
        test_lap = calculate_laplacian_variance(test_frame)

        if clean_lap > 0:
            laplacian_ratio = test_lap / clean_lap
            sharpness_drop_percent = (1.0 - laplacian_ratio) * 100.0
        else:
            laplacian_ratio = None
            sharpness_drop_percent = None

        timestamp_sec = frame_index / clean_fps if clean_fps > 0 else None

        results.append({
            "frame_index": frame_index,
            "timestamp_sec": timestamp_sec,
            "clean_laplacian_variance": clean_lap,
            "test_laplian_variance": test_lap,
            "laplacian_ratio": laplacian_ratio,
            "sharpness_drop_percent": sharpness_drop_percent
        })

        frame_index += 1

        if frame_index % 100 == 0:
            print(f"已处理 {frame_index}/{total_frames} 帧")

    clean_cap.release()
    test_cap.release()

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame_index",
            "timestamp_sec",
            "clean_laplacian_variance",
            "test_laplian_variance",
            "laplacian_ratio",
            "sharpness_drop_percent"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    clean_values = np.array(
        [r["clean_laplacian_variance"] for r in results],
        dtype=np.float64
    )

    test_values = np.array(
        [r["test_laplian_variance"] for r in results],
        dtype=np.float64
    )

    ratio_values = np.array(
        [
            r["laplacian_ratio"]
            for r in results
            if r["laplacian_ratio"] is not None
        ],
        dtype=np.float64
    )

    print("处理完成")
    print(f"实际比较帧数: {frame_index}")
    print(f"结果已保存到: {output_csv}")

    if len(clean_values) > 0 and len(test_values) > 0:
        print(f"干净视频平均清晰度: {clean_values.mean():.2f}")
        print(f"测试视频平均清晰度: {test_values.mean():.2f}")

    if len(ratio_values) > 0:
        print(f"平均清晰度比例 test/clean: {ratio_values.mean():.4f}")
        print(f"平均清晰度下降: {(1.0 - ratio_values.mean()) * 100.0:.2f}%")
        print(f"最小清晰度比例: {ratio_values.min():.4f}")
        print(f"最大清晰度比例: {ratio_values.max():.4f}")

def save_first_frame(video_path, output_image_path):
    video_path = Path(video_path)
    output_image_path = Path(output_image_path)

    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("无法读取视频第一帧")

    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_image_path), frame)

    if not success:
        raise RuntimeError(f"保存图片失败: {output_image_path}")

    print(f"第一帧已保存到: {output_image_path}")


def main():
    # analyze_video_sharpness(
    #     '/home/wjh/Wan2.2/opens2v_outputs/test/i2v_svdquant_all_precision.mp4',
    #     '/home/wjh/Wan2.2/opens2v_outputs/test/i2v_svdquant_gptq.mp4',
    #     "sharpness_compare.csv"
    # )
    save_first_frame('/home/wjh/Wan2.2/opens2v_outputs/test/i2v_svdquant_gptq.mp4', "first_frame.jpg")


if __name__ == "__main__":
    main()