#!/usr/bin/env python3
"""
从 HuggingFace 下载 LeRobot 数据集并展示视频。

用法:
    python view_dataset_video.py

功能:
  1. 从 HuggingFace 下载数据集到本地缓存
  2. 将所有摄像头的 AV1 视频拼接成一个 h264 mp4 文件
  3. 输出文件路径，可直接用播放器打开

注意: 原始视频使用 AV1 编码，通过 pyav 解码后重新编码为 h264。
"""

import sys
import time
from pathlib import Path

import av
import numpy as np

# ─── 配置 ────────────────────────────────────────────────
REPO_ID = "FrankYuzhe/lemon_box_right_front_40_20260226_185838"
EPISODES = [0]           # 要查看的 episode 列表，None 表示全部
OUTPUT_DIR = "/home/robotlab/lerobot/dataset_videos"  # 输出目录
# ──────────────────────────────────────────────────────────


def main():
    print(f"📦 正在加载数据集: {REPO_ID}")
    print(f"   这可能需要一些时间（首次会从 HuggingFace 下载）...")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        episodes=EPISODES,
    )

    print(f"✅ 数据集加载完成!")
    print(f"   总帧数: {dataset.num_frames}")
    print(f"   总 episodes: {dataset.num_episodes}")
    print(f"   FPS: {dataset.fps}")
    print(f"   摄像头: {dataset.meta.camera_keys}")
    print(f"   视频 keys: {dataset.meta.video_keys}")
    print(f"   图像 keys: {dataset.meta.image_keys}")
    print(f"   所有 features: {list(dataset.meta.features.keys())}")
    print()

    video_keys = dataset.meta.video_keys
    if not video_keys:
        print("❌ 数据集中没有视频数据!")
        return

    # 打印原始视频文件位置
    print("📁 原始视频文件位置:")
    for ep_idx in (dataset.episodes if dataset.episodes else range(dataset.num_episodes)):
        for vk in video_keys:
            try:
                vpath = dataset.root / dataset.meta.get_video_file_path(ep_idx, vk)
                if vpath.exists():
                    print(f"   Episode {ep_idx} / {vk}: {vpath}")
            except Exception:
                pass
    print()

    # 生成拼接后的视频
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in (dataset.episodes if dataset.episodes else range(dataset.num_episodes)):
        _export_combined_video(dataset, video_keys, ep_idx, output_dir)

    print("\n🏁 全部完成!")


def _export_combined_video(dataset, video_keys, ep_idx, output_dir):
    """将多个摄像头视频拼接成一个 h264 mp4 文件"""
    fps = dataset.fps
    repo_name = dataset.repo_id.replace("/", "_")
    output_path = output_dir / f"{repo_name}_ep{ep_idx:03d}_combined.mp4"

    print(f"🎬 处理 Episode {ep_idx}...")

    # 打开所有视频
    containers = {}
    for vk in video_keys:
        try:
            vpath = dataset.root / dataset.meta.get_video_file_path(ep_idx, vk)
            if vpath.exists():
                containers[vk] = av.open(str(vpath))
                stream = containers[vk].streams.video[0]
                print(f"   📹 {vk}: {stream.width}x{stream.height}, codec={stream.codec_context.name}")
        except Exception as e:
            print(f"   ⚠️  {vk}: 打开失败 - {e}")

    if not containers:
        print("   ❌ 没有可处理的视频!")
        return

    # 创建帧迭代器
    frame_iters = {vk: c.decode(video=0) for vk, c in containers.items()}

    # 先读取第一帧确定尺寸
    first_frames = {}
    for vk in list(frame_iters.keys()):
        try:
            f = next(frame_iters[vk])
            first_frames[vk] = f.to_ndarray(format='rgb24')
        except StopIteration:
            del frame_iters[vk]

    if not first_frames:
        print("   ❌ 无法读取帧!")
        return

    # 计算拼接后的尺寸（横向拼接）
    target_h = max(img.shape[0] for img in first_frames.values())
    total_w = 0
    for img in first_frames.values():
        scale = target_h / img.shape[0]
        total_w += int(img.shape[1] * scale)

    # 确保宽高是偶数（h264 要求）
    total_w = total_w + (total_w % 2)
    target_h = target_h + (target_h % 2)

    print(f"   📐 输出尺寸: {total_w}x{target_h}")

    # 创建输出视频
    out_container = av.open(str(output_path), mode='w')
    out_stream = out_container.add_stream('h264', rate=fps)
    out_stream.width = total_w
    out_stream.height = target_h
    out_stream.pix_fmt = 'yuv420p'
    # 设置较高质量
    out_stream.options = {'crf': '18', 'preset': 'fast'}

    frame_count = 0

    def compose_frame(frame_dict):
        """将多个摄像头帧拼接"""
        imgs = []
        for vk in video_keys:
            if vk in frame_dict:
                img = frame_dict[vk]
                if img.shape[0] != target_h:
                    scale = target_h / img.shape[0]
                    new_w = int(img.shape[1] * scale)
                    # 使用简单的最近邻缩放
                    from PIL import Image
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize((new_w, target_h), Image.BILINEAR)
                    img = np.array(pil_img)
                imgs.append(img)

        if not imgs:
            return None

        combined = np.hstack(imgs)
        # 确保尺寸匹配
        if combined.shape[1] != total_w or combined.shape[0] != target_h:
            from PIL import Image
            pil_img = Image.fromarray(combined)
            pil_img = pil_img.resize((total_w, target_h), Image.BILINEAR)
            combined = np.array(pil_img)

        # 添加标签
        _add_labels(combined, list(frame_dict.keys()), frame_count, ep_idx, fps)

        return combined

    # 写入第一帧
    combined = compose_frame(first_frames)
    if combined is not None:
        out_frame = av.VideoFrame.from_ndarray(combined, format='rgb24')
        for packet in out_stream.encode(out_frame):
            out_container.mux(packet)
        frame_count += 1

    # 写入剩余帧
    while frame_iters:
        current_frames = {}
        for vk in list(frame_iters.keys()):
            try:
                f = next(frame_iters[vk])
                current_frames[vk] = f.to_ndarray(format='rgb24')
            except StopIteration:
                del frame_iters[vk]

        if not current_frames:
            break

        combined = compose_frame(current_frames)
        if combined is not None:
            out_frame = av.VideoFrame.from_ndarray(combined, format='rgb24')
            for packet in out_stream.encode(out_frame):
                out_container.mux(packet)
            frame_count += 1

        if frame_count % 100 == 0:
            print(f"   ⏳ 已处理 {frame_count} 帧...")

    # 刷新编码器
    for packet in out_stream.encode():
        out_container.mux(packet)

    out_container.close()
    for c in containers.values():
        c.close()

    print(f"   ✅ 完成! {frame_count} 帧已保存")
    print(f"   📄 输出文件: {output_path}")
    print(f"   💡 播放命令: ffplay {output_path}")

    # 同时也单独导出每个摄像头的 h264 版本
    print(f"\n   📹 同时导出各摄像头独立视频...")
    for vk in video_keys:
        try:
            vpath = dataset.root / dataset.meta.get_video_file_path(ep_idx, vk)
            if vpath.exists():
                cam_name = vk.replace(".", "_")
                single_output = output_dir / f"{repo_name}_ep{ep_idx:03d}_{cam_name}.mp4"
                _transcode_to_h264(vpath, single_output, fps)
                print(f"      ✅ {vk} -> {single_output}")
        except Exception as e:
            print(f"      ⚠️  {vk}: 转码失败 - {e}")


def _transcode_to_h264(input_path, output_path, fps):
    """将 AV1 视频转码为 h264"""
    in_container = av.open(str(input_path))
    out_container = av.open(str(output_path), mode='w')

    in_stream = in_container.streams.video[0]
    out_stream = out_container.add_stream('h264', rate=fps)
    out_stream.width = in_stream.width
    out_stream.height = in_stream.height
    out_stream.pix_fmt = 'yuv420p'
    out_stream.options = {'crf': '18', 'preset': 'fast'}

    for frame in in_container.decode(video=0):
        out_frame = av.VideoFrame.from_ndarray(frame.to_ndarray(format='rgb24'), format='rgb24')
        for packet in out_stream.encode(out_frame):
            out_container.mux(packet)

    for packet in out_stream.encode():
        out_container.mux(packet)

    out_container.close()
    in_container.close()


def _add_labels(img, labels, frame_idx, ep_idx, fps):
    """在图像上添加文字标签（使用 PIL 避免 OpenCV GUI 依赖）"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        # 尝试加载字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()
            font_small = font

        # 摄像头标签
        x_offset = 10
        for label in labels:
            # 背景
            bbox = draw.textbbox((x_offset, 8), label, font=font)
            draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=(0, 0, 0, 180))
            draw.text((x_offset, 8), label, fill=(0, 255, 0), font=font)
            x_offset += img.shape[1] // len(labels)

        # 帧信息
        info_text = f"Episode: {ep_idx} | Frame: {frame_idx} | FPS: {fps}"
        h = img.shape[0]
        bbox = draw.textbbox((10, h - 28), info_text, font=font_small)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=(0, 0, 0, 180))
        draw.text((10, h - 28), info_text, fill=(255, 255, 255), font=font_small)

        # 写回 numpy 数组
        result = np.array(pil_img)
        img[:] = result[:]
    except ImportError:
        pass  # 没有 PIL 就跳过标签


if __name__ == "__main__":
    main()
