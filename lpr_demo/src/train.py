#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/4/23 20:43
@version : 1.0.0
@author  : William_Trouvaille
@function: 训练代码
"""
# TODO，见Markdown笔记
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, DataLoader, random_split

from src.utils import Loader, LoggerHandler, TrainingProgress


class LPRDataset(Dataset):
    """车牌识别数据集类"""

    def __init__(self,
                 data_cfg: Path,
                 img_size: int = 640,
                 augment: bool = True):
        """
        Args:
            data_cfg: YOLO格式数据集配置文件路径
            img_size: 输入图像尺寸
            augment: 是否启用数据增强
        """
        # 解析YOLO数据集配置
        with open(data_cfg, 'r') as f:
            self.data_info = yaml.safe_load(f)

        self.img_files = self._load_image_paths()
        self.img_size = img_size
        self.augment = augment

    def _load_image_paths(self) -> List[Path]:
        """严格遵循项目结构的路径加载方法"""
        valid_files = []
        for split in ['train', 'val', 'test']:
            # labels目录路径
            label_split_dir = Loader.get_path("/data/labels") / split

            # 遍历所有标签文件
            for label_file in label_split_dir.glob('*.txt'):
                # 生成对应的图像路径
                img_name = label_file.stem + '.jpg'  # 关键修正点：正确获取图片文件名
                img_split_dir = Loader.get_path("/data/images") / split
                img_path = img_split_dir / img_name

                # 双重验证路径有效性
                if not img_path.exists():
                    LoggerHandler().error(f"Image {img_path} not found for label {label_file}")
                    continue
                if not label_file.exists():
                    LoggerHandler().error(f"Label {label_file} not found for image {img_path}")
                    continue

                valid_files.append(img_path)

        LoggerHandler().info(f"Loaded {len(valid_files)} valid image-label pairs")
        return valid_files

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # 加载图像
        img_path = self.img_files[idx]
        try:
            # 检查文件是否存在
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

            # 读取图像并验证
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Failed to read image: {img_path}")

            # 转换颜色空间
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 检查图像尺寸
            if img.size == 0:
                raise ValueError(f"Empty image: {img_path}")

        except Exception as e:
            # 记录错误并跳过该样本
            LoggerHandler().error(str(e))
            return self.__getitem__((idx + 1) % len(self))  # 跳过错误样本

        # 加载标注
        label_path = img_path.parent.parent / 'labels' / img_path.name.replace('.jpg', '.txt')
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # 解析检测框（首行）
        det_line = lines[0].strip().split()
        bbox = list(map(float, det_line[1:5]))

        # 解析车牌号（次行）
        plate_number = lines[1].strip()

        # 数据增强（示例实现）
        if self.augment:
            img, bbox = self._random_horizontal_flip(img, bbox)
            img = self._color_jitter(img)

        # 转换为Tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        target = {
            'detection': torch.tensor(bbox),
            'recognition': plate_number
        }

        return img, target


class MultiTaskModel(nn.Module):
    """YOLO+CRNN联合模型"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self._build_detector()
        self._build_recognizer()

    def _build_detector(self):
        """初始化YOLO检测网络"""
        from ultralytics import YOLO
        # 加载预训练配置
        cfg = self.config['model']['detector']
        self.detector = YOLO(cfg['pretrained']).model  # 直接获取底层模型

        # 修正参数冻结方式
        if cfg.get('freeze_backbone', False):
            # 冻结backbone前10层参数
            for name, param in self.detector.named_parameters():
                if 'model.0' in name:  # backbone通常对应model[0]
                    param.requires_grad = False

    def _build_recognizer(self):
        """初始化CRNN识别网络"""
        all_chars = (
                self.config['provinces'] +
                self.config['alphabets'] +
                self.config['ads']
        )
        unique_chars = sorted(list(set(all_chars)))

        self.recognizer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.LSTM(768, 256, bidirectional=True, batch_first=True),
            nn.Linear(512, len(unique_chars) + 1)  # 包含空白符
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        det_output = self.detector(x)
        crops = self._crop_plates(x, det_output)
        recog_output = self.recognizer(crops)
        return det_output, recog_output

    def _crop_plates(self, x: torch.Tensor, detections: torch.Tensor) -> torch.Tensor:
        """根据检测结果裁剪车牌区域"""
        # 实现细节需根据实际检测输出结构调整
        pass


class LPRTrainer:
    """多任务训练控制器"""

    def __init__(self):
        self.progress = None
        self.config = Loader.load_config()
        self.logger = LoggerHandler()
        self.device = self._init_device()
        self.model = self._init_model()
        self._init_paths()

    def _init_device(self) -> str:
        """自动选择训练设备"""
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def _init_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        # 创建完整数据集并验证有效性
        full_dataset = LPRDataset(
            data_cfg=Loader.get_path('data/yolo/ccpd.yaml'),
            img_size=self.config['model']['recognizer']['input_size']
        )

        if len(full_dataset) == 0:
            raise RuntimeError("Dataset is empty. Check data paths and file integrity.")

        # 计算划分比例
        val_split = self.config['training']['validation_split']
        if not (0 < val_split < 1):
            raise ValueError(f"Invalid validation split: {val_split}. Must be between 0 and 1.")

        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size

        # 验证划分结果
        if train_size <= 0 or val_size <= 0:
            raise RuntimeError(
                f"Invalid split sizes: train={train_size}, val={val_size}. "
                f"Total samples: {len(full_dataset)}"
            )

        # 执行数据划分
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # 记录数据集信息
        self.logger.info(
            f"Dataset split | Train: {len(train_dataset)} "
            f"Val: {len(val_dataset)}"
        )

        return (
            self._create_loader(train_dataset, shuffle=True),
            self._create_loader(val_dataset, shuffle=False)
        )

    def _create_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """创建数据加载器"""
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=shuffle,
            num_workers=self.config['training'].get('num_workers', 2),
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate_fn  # 新增数据打包函数
        )

    @staticmethod
    def _collate_fn(batch):
        """自定义数据打包逻辑"""
        images, targets = zip(*batch)
        return torch.stack(images), targets

    def _init_model(self) -> MultiTaskModel:
        """初始化联合模型"""
        model = MultiTaskModel(self.config)
        model.to(self.device)

        # 加载预训练权重
        if self.config['model'].get('pretrained'):
            state_dict = torch.load(Loader.get_path(self.config['model']['pretrained']))
            model.load_state_dict(state_dict)

        return model

    def _init_paths(self):
        """初始化路径配置"""
        self.output_dir = Loader.get_path('models/')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建当前训练会话目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / timestamp
        self.session_dir.mkdir()

    def _init_optimizers(self) -> Dict:
        """初始化优化器"""
        det_params = filter(lambda p: p.requires_grad, self.model.detector.parameters())
        recog_params = self.model.recognizer.parameters()

        return {
            'detector': torch.optim.Adam(det_params, lr=self.config['optim']['det_lr']),
            'recognizer': torch.optim.SGD(recog_params, lr=self.config['optim']['recog_lr'])
        }

    def train(self):
        """执行完整训练流程"""
        self.logger.info("Initializing training session...")

        optimizers = self._init_optimizers()
        train_loader, val_loader = self._init_dataloaders()
        best_metric = float('inf')
        no_improve_epochs = 0
        self.progress = TrainingProgress(  # 赋值给实例变量
            total_epochs=self.config['training']['epochs'],
            train_loader_len=len(train_loader),
            val_loader_len=len(val_loader),
            metrics=['det_loss', 'recog_loss', 'total_loss']
        )

        # 初始化进度跟踪
        progress = TrainingProgress(
            total_epochs=self.config['training']['epochs'],
            train_loader_len=len(train_loader),
            val_loader_len=len(val_loader),
            metrics=['det_loss', 'recog_loss', 'total_loss']
        )

        for epoch in range(self.config['training']['epochs']):
            epoch_progress = progress.epoch_progress(epoch + 1)

            # 训练阶段
            train_metrics = self._run_epoch(train_loader,
                                            optimizers,
                                            is_train=True,
                                            current_epoch=epoch + 1)

            # 验证阶段
            with torch.no_grad():
                val_metrics = self._run_epoch(val_loader, None, is_train=False, current_epoch=epoch + 1)

            # 记录最佳模型
            if val_metrics['total_loss'] < best_metric:
                self._save_checkpoint(epoch, val_metrics)
                best_metric = val_metrics['total_loss']
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # 早停判断
            if no_improve_epochs >= self.config['training']['patience']:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            # 更新进度
            epoch_progress.set_postfix({
                'det_loss': f"{train_metrics['det_loss']:.3f}/{val_metrics['det_loss']:.3f}",
                'recog_loss': f"{train_metrics['recog_loss']:.3f}/{val_metrics['recog_loss']:.3f}",
                'total': f"{train_metrics['total_loss']:.3f}/{val_metrics['total_loss']:.3f}"
            })
            epoch_progress.update(1)

    def _run_epoch(self,
                   loader,
                   optimizers: Dict,
                   is_train: bool,
                   current_epoch: int) -> Dict:
        """执行单个epoch"""
        self.model.train() if is_train else self.model.eval()
        metrics = {'det_loss': 0.0, 'recog_loss': 0.0, 'total_loss': 0.0}

        batch_progress = self.progress.batch_progress(
            mode='TRAIN' if is_train else 'VAL',
            current_epoch=current_epoch + 1
        )

        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(self.device)
            det_targets = targets['detection'].to(self.device)
            recog_targets = targets['recognition'].to(self.device)

            # 前向传播
            det_output, recog_output = self.model(images)

            # 计算损失
            det_loss = self._detection_loss(det_output, det_targets)
            recog_loss = self._recognition_loss(recog_output, recog_targets)
            total_loss = det_loss + recog_loss * self.config['loss']['recog_weight']

            # 反向传播
            if is_train:
                for opt in optimizers.values():
                    opt.zero_grad()
                total_loss.backward()
                for opt in optimizers.values():
                    opt.step()

            # 记录指标
            metrics['det_loss'] += det_loss.item()
            metrics['recog_loss'] += recog_loss.item()
            metrics['total_loss'] += total_loss.item()

            # 更新进度
            if batch_idx % 10 == 0:
                batch_progress.set_postfix({
                    'det': f"{det_loss.item():.3f}",
                    'recog': f"{recog_loss.item():.3f}"
                })
            batch_progress.update(1)

        # 计算平均损失
        for k in metrics:
            metrics[k] /= len(loader)

        return metrics

    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'metrics': metrics
        }
        path = self.session_dir / f"best_{epoch}.pt"
        torch.save(checkpoint, path)
        self.logger.info(f"Saved best model at epoch {epoch} with loss {metrics['total_loss']:.4f}")


def main():
    trainer = LPRTrainer()
    try:
        trainer.train()
    except Exception as e:
        trainer.logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
