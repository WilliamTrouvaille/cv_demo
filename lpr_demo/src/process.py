#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/4/21 23:33
@version : 1.0.0
@author  : William_Trouvaille
@function: 数据预处理
"""

# TODO：①将原始数据集转换为YOLO格式；
#  ②使用透视变化提取出车牌；
#  ③压缩成果文件并将其上传git
import os
import shutil
import traceback
from pathlib import Path
from typing import Tuple, List
import platform

import yaml
from PIL import Image

from src.utils import Loader, LoggerHandler, TrainingProgress

logger = LoggerHandler()



class CCPDProcessor:
    """CCPD数据集处理工具类"""

    def __init__(self):
        self.config = Loader.load_config()
        # self.logger = LoggerHandler()
        self._validate_config()
        self._init_paths()

    def _validate_config(self):
        """验证配置文件完整性"""
        required = ['data_paths', 'provinces', 'alphabets', 'ads']
        for key in required:
            if key not in self.config:
                raise KeyError(f"Missing required config key: {key}")

    def _init_paths(self):
        """初始化路径配置"""
        # 原始数据路径
        self.raw_root = Loader.get_path(self.config['data_paths']['raw'])

        # YOLO格式路径
        self.yolo_root = Loader.get_path(self.config['data_paths']['yolo'])
        self.yolo_images = self.yolo_root / 'images'
        self.yolo_labels = self.yolo_root / 'labels'

        # 创建必要目录
        self.yolo_images.mkdir(parents=True, exist_ok=True)
        self.yolo_labels.mkdir(parents=True, exist_ok=True)

    def parse_filename(self, filename: str) -> Tuple[List[Tuple[int, int]], str]:
        """
        解析CCPD文件名，返回顶点坐标和车牌号
        格式：015-91_90-248&454_464&524-464&524_248&520_248&454_462&460-0_0_3_24_33_29_27_27-141-187
        """
        parts = filename.split('-')
        if len(parts) < 7:
            raise ValueError(f"Invalid filename format: {filename}")

        # 解析顶点坐标（第四部分）
        coord_part = parts[3]
        vertices = [tuple(map(int, p.split('&'))) for p in coord_part.split('_')]
        if len(vertices) != 4:
            raise ValueError(f"Invalid vertices count: {len(vertices)}")

        # 解析车牌号（第五部分）
        code_part = parts[4].split('_')
        province = self.config['provinces'][int(code_part[0])]
        alphabet = self.config['alphabets'][int(code_part[1])]
        ad_chars = [self.config['ads'][int(c)] for c in code_part[2:7]]
        plate_number = f"{province}{alphabet}-{''.join(ad_chars)}"

        return vertices, plate_number

    def vertices_to_yolo(self,
                         vertices: List[Tuple[int, int]],
                         img_w: int,
                         img_h: int) -> Tuple[float, float, float, float]:
        """将顶点坐标转换为YOLO格式的归一化边界框"""
        x_coords = [x for x, _ in vertices]
        y_coords = [y for _, y in vertices]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        x_center = (x_min + x_max) / 2 / img_w
        y_center = (y_min + y_max) / 2 / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h

        return x_center, y_center, width, height

    def _create_file_link(self, src: Path, dst: Path):
        """创建文件链接（跨平台兼容）"""
        if dst.exists():
            return

        if Loader.is_windows():
            shutil.copy(src, dst)
        else:
            os.symlink(src.resolve(), dst)

    def process_dataset(self, split: str):
        """处理指定数据集划分"""
        # 初始化路径
        raw_split_dir = self.raw_root / split
        yolo_img_dir = self.yolo_images / split
        yolo_label_dir = self.yolo_labels / split

        # 创建目标目录
        yolo_img_dir.mkdir(parents=True, exist_ok=True)
        yolo_label_dir.mkdir(parents=True, exist_ok=True)

        # 获取文件列表
        img_files = list(raw_split_dir.glob('*.jpg'))
        total = len(img_files)

        # 初始化进度条
        progress = TrainingProgress(
            total_epochs=1,
            train_loader_len=total,
            metrics=['processed', 'errors']
        )
        pbar = progress.batch_progress(mode=split.upper())

        processed = 0
        errors = 0

        for img_path in img_files:
            try:
                # 创建图片链接
                link_path = yolo_img_dir / img_path.name
                self._create_file_link(img_path, link_path)

                # 获取图片尺寸
                with Image.open(img_path) as img:
                    img_w, img_h = img.size

                # 解析文件名
                filename = img_path.stem
                vertices, plate_num = self.parse_filename(filename)

                # 生成YOLO标注
                bbox = self.vertices_to_yolo(vertices, img_w, img_h)
                label_path = yolo_label_dir / f"{filename}.txt"
                with open(label_path, 'w') as f:
                    f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                processed += 1
            except Exception as e:
                errors += 1
                logger.error(f"Process {img_path.name} failed: {str(e)}")
                if self.config.get('debug', False):
                    traceback.print_exc()

            # 更新进度
            progress.update_metrics(pbar, {
                'processed': processed,
                'errors': errors
            })
            pbar.update(1)

        pbar.close()
        logger.info(
            f"{split.upper()} Process Report | "
            f"Total: {total}, Success: {processed}, Errors: {errors}, "
            f"Success Rate: {(processed / total) * 100:.2f}%"
        )

    def generate_data_yaml(self):
        """生成YOLO数据集配置文件"""
        data = {
            'path': str(self.yolo_root.resolve()),
            'train': 'train.txt',
            'val': 'val.txt',
            'test': 'test.txt',
            'names': {0: 'license_plate'},
            'nc': 1
        }

        yaml_path = self.yolo_root / 'ccpd.yaml'
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)

        logger.info(f"YOLO dataset config generated at {yaml_path}")


def main():
    logger.info("Starting CCPD dataset processing...")
    processor = CCPDProcessor()

    # 处理所有数据集划分
    for split in ['train', 'val', 'test']:
        if (processor.raw_root / split).exists():
            processor.process_dataset(split)

    # 生成YOLO配置文件
    processor.generate_data_yaml()
    logger.info("All processing completed!")


if __name__ == "__main__":
    main()
