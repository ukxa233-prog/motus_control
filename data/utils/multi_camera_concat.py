#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import argparse
import os
from typing import Optional, Tuple

def resize_and_concatenate_frames(
    head_img: np.ndarray, 
    left_img: np.ndarray, 
    right_img: np.ndarray
) -> Optional[np.ndarray]:
    """
    将三个相机视角拼接为 T 字型布局：
    - 上方：头部相机 (保持原大，如 480x640)
    - 下左：左手腕相机 (缩小至 1/2，如 240x320)
    - 下右：右手腕相机 (缩小至 1/2，如 240x320)
    最终输出：720x640 (高 x 宽)
    """
    try:
        # 获取原始维度
        orig_h, orig_w = head_img.shape[:2]
            
        # 将手腕相机缩放为一半大小
        half_h, half_w = orig_h // 2, orig_w // 2
        
        # 确保缩放后的宽度之和等于头部相机宽度
        # 使用 (half_w, half_h) 因为 cv2.resize 接收 (宽, 高)
        left_resized = cv2.resize(left_img, (half_w, half_h))
        right_resized = cv2.resize(right_img, (orig_w - half_w, half_h)) 
            
        # 水平拼接手腕相机图像作为底行
        bottom_row = np.hstack([left_resized, right_resized])
            
        # 垂直拼接：顶部头部相机 + 底部手腕行
        combined = np.vstack([head_img, bottom_row])
            
        return combined
    except Exception as e:
        print(f"拼接失败: {e}")
        return None

def get_concatenated_dimensions(original_shape: Tuple[int, int]) -> Tuple[int, int]:
    """计算拼接后的输出维度 (高, 宽)"""
    h, w = original_shape
    # 最终高度 = 原高 + 0.5*原高 = 1.5h; 宽度不变
    return int(h * 1.5), w

def main():
    parser = argparse.ArgumentParser(description="多摄像头视角拼接工具")
    parser.add_argument("--head_image", required=True, help="头部相机图片路径")
    parser.add_argument("--left_image", required=True, help="左手腕相机图片路径")
    parser.add_argument("--right_image", required=True, help="右手腕相机图片路径")
    parser.add_argument("--output", required=True, help="拼接结果保存路径")
    
    args = parser.parse_args()

    # 读取图像
    head = cv2.imread(args.head_image)
    left = cv2.imread(args.left_image)
    right = cv2.imread(args.right_image)

    if head is None or left is None or right is None:
        print("错误: 无法读取输入图片，请检查路径。")
        return

    # 执行拼接
    result = resize_and_concatenate_frames(head, left, right)
    
    if result is not None:
        # 确保输出目录存在
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        cv2.imwrite(args.output, result)
        print(f"成功保存拼接图至: {args.output}")
        print(f"结果尺寸: {result.shape} (高, 宽, 通道)")
    else:
        print("处理失败。")

if __name__ == "__main__":
    main()