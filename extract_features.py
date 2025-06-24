import os
import sys
import argparse
import numpy as np
import torch
import torch.utils.data
from torchvision.transforms import Compose
import cv2
from tqdm import tqdm

import videotransforms
from pytorch_i3d import InceptionI3d

def load_video_frames(video_path):
    """
    使用OpenCV从视频文件中加载所有帧。
    Args:
        video_path (str): 视频文件的路径。
    Returns:
        np.ndarray: 一个形状为 (T, H, W, C) 的Numpy数组，其中T是帧数，或在失败时返回None。
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return np.array(frames)
    except Exception:
        return None

class VideoDatasetForFeatureExtraction(torch.utils.data.Dataset):
    def __init__(self, video_dir, output_dir, transforms):
        self.video_dir = video_dir
        self.transforms = transforms
        
        all_video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.mkv', '.avi'))]
        
        self.video_files_to_process = []
        for video_name in all_video_files:
            file_name, _ = os.path.splitext(video_name)
            output_path = os.path.join(output_dir, f"{file_name}.npy")
            if not os.path.exists(output_path):
                self.video_files_to_process.append(video_name)

    def __len__(self):
        return len(self.video_files_to_process)

    def __getitem__(self, idx):
        video_name = self.video_files_to_process[idx]
        video_path = os.path.join(self.video_dir, video_name)
        
        video_frames = load_video_frames(video_path)
        
        if video_frames is None or len(video_frames) == 0:
            return "ERROR", video_name
            
        video_transformed = self.transforms(video_frames)
        
        video_tensor = torch.from_numpy(video_transformed).permute(3, 0, 1, 2)
        video_tensor = (video_tensor / 255.0) * 2.0 - 1.0
        
        return video_tensor, video_name

def collate_fn_separate_errors(batch):
    """
    一个自定义的collate_fn，它将成功加载的样本和加载失败的样本分离开。
    加载失败的样本在__getitem__中返回了 ('ERROR', video_name)。
    """
    good_samples = []
    error_video_names = []
    for item in batch:
        # item 是一个元组, e.g., (tensor, video_name) or ("ERROR", video_name)
        if item[0] == "ERROR":
            error_video_names.append(item[1])
        else:
            good_samples.append(item)
    
    # 如果这个批次中所有样本都加载失败
    if not good_samples:
        return None, error_video_names
    
    # 正常处理好的样本
    collated_good_samples = torch.utils.data.default_collate(good_samples)
    return collated_good_samples, error_video_names

def run(video_dir, output_dir, load_model, mode, gpu, num_workers):
    """
    从指定目录的视频中提取I3D特征，并将其保存为.npy文件。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    if not os.path.exists(output_dir):
        print(f"输出目录 {output_dir} 不存在，正在创建...")
        os.makedirs(output_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    print("正在加载I3D模型...")
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    
    print(f"从 {load_model} 加载模型权重...")
    try:
        i3d.load_state_dict(torch.load(load_model, map_location=device, weights_only=True))
    except Exception as e:
        print(f"错误：无法加载模型权重，请检查路径和文件是否正确: {e}")
        return
        
    i3d.to(device)
    i3d.eval()

    test_transforms = Compose([videotransforms.CenterCrop(224)])

    # --- 修改：提供更清晰的初始报告 ---
    all_video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.mkv', '.avi'))]
    existing_feature_files = {os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.endswith('.npy')}
    
    files_to_process = [f for f in all_video_files if os.path.splitext(f)[0] not in existing_feature_files]

    print("--- 任务初始报告 ---")
    print(f"视频总数: {len(all_video_files)}")
    print(f"已提取特征数 (将跳过): {len(existing_feature_files)}")
    print(f"本次需新提取特征数: {len(files_to_process)}")
    print("--------------------")

    if not files_to_process:
        print("所有视频的特征均已提取完毕。")
        return
        
    # --- 修改：现在只将需要处理的文件传递给Dataset ---
    dataset = VideoDatasetForFeatureExtraction(video_dir, output_dir, test_transforms)
    # 更新了collate_fn
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_separate_errors
    )
    
    # --- 修改：添加计数器用于最终报告 ---
    successfully_processed_count = 0
    failed_count = 0
    
    # --- 修改：将日志文件路径定义到项目根目录 ---
    log_file_path = 'failed_videos.log'

    # --- 修改：使用 with open 确保日志文件被妥善处理 ---
    with open(log_file_path, 'a') as log_file:
        progress_bar = tqdm(dataloader, desc="正在提取特征")
        for data in progress_bar:
            # collate_fn现在返回 (collated_good_samples, error_video_names)
            collated_good_samples, error_video_names = data

            # --- 修改：实时报告失败的文件到日志 ---
            if error_video_names:
                for video_name in error_video_names:
                    # 移除原有的tqdm.write，改为写入文件
                    # --- 修改：修正换行符 ---
                    log_file.write(f"{video_name}\n")
                    log_file.flush() # --- 新增：确保实时写入磁盘 ---
                    failed_count += 1

            if collated_good_samples is None:
                # 更新进度条后继续
                progress_bar.set_postfix_str(f"成功: {successfully_processed_count} | 失败: {failed_count}")
                continue
            
            video_tensor, video_name_tuple = collated_good_samples
            video_name = video_name_tuple[0]

            file_name, _ = os.path.splitext(video_name)
            output_path = os.path.join(output_dir, f"{file_name}.npy")
            
            video_tensor = video_tensor.to(device, dtype=torch.float32)

            with torch.no_grad():
                features = i3d.extract_features(video_tensor)

            features_np = features.squeeze(0).cpu().numpy()
            np.save(output_path, features_np)
            successfully_processed_count += 1
            
            # --- 新增：在每次迭代后更新进度条的后缀 ---
            progress_bar.set_postfix_str(f"成功: {successfully_processed_count} | 失败: {failed_count}")
        
    # --- 新增：提供最终总结报告 ---
    print("\n--- 特征提取完成 ---")
    print(f"本次运行成功提取: {successfully_processed_count} 个新文件")
    print(f"本次运行读取失败: {failed_count} 个文件")
    final_feature_count = len([name for name in os.listdir(output_dir) if name.endswith(".npy")])
    print(f"输出目录中总计: {final_feature_count} 个特征文件")
    print("--------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从视频中提取I3D特征并保存为.npy文件。")
    parser.add_argument('--video_dir', type=str, required=True, help='包含视频文件的目录。')
    parser.add_argument('--output_dir', type=str, required=True, help='用于保存.npy特征文件的目录。')
    parser.add_argument('--load_model', type=str, required=True, help='预训练I3D模型权重(.pt文件)的路径。')
    parser.add_argument('--mode', type=str, default='rgb', choices=['rgb', 'flow'], help="模型模式，本项目中应为'rgb'。")
    parser.add_argument('--gpu', type=str, default='0', help="要使用的GPU设备ID，例如'0'。")
    parser.add_argument('--num_workers', type=int, default=4, help="数据加载使用的工作进程数。建议设置为CPU核心数的一半。")

    args = parser.parse_args()

    run(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        load_model=args.load_model,
        mode=args.mode,
        gpu=args.gpu,
        num_workers=args.num_workers
    )
