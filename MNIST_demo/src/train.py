import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.utils import Loader, LoggerHandler, TrainingProgress


class CNN(nn.Module):
    """简单的CNN网络结构，用于MNIST分类"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class MNISTTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = LoggerHandler()
        self._init_components()
        self.best_acc = 0.0
        self.early_stop_counter = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def _init_components(self):
        """初始化数据加载、模型、优化器等组件"""
        # 数据预处理
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 数据集加载
        train_set = torchvision.datasets.MNIST(
            root=Loader.get_path("/data"),
            train=True,
            download=True,
            transform=transform
        )
        val_set = torchvision.datasets.MNIST(
            root=Loader.get_path("/data"),
            train=False,
            download=True,
            transform=transform
        )

        # 数据加载器
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=2
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=2
        )

        # 模型初始化
        self.model = CNN().to(self.device)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=2,
            factor=0.5,
            verbose=False
        )

    def _save_model(self, epoch, is_best=False):
        """模型保存逻辑"""
        model_path = Loader.build_path(self.config['paths']['model_save'])
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc
        }, model_path)
        if is_best:
            self.logger.info(f"** 最佳模型已保存至 {model_path}，准确率 {self.best_acc:.4f} **")

    def _train_epoch(self, epoch, progress):
        """单个epoch的训练逻辑"""
        self.model.train()
        train_loss, correct, total = 0.0, 0, 0

        batch_progress = progress.batch_progress('train', epoch)
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            batch_progress.update(1)
            progress.update_metrics(batch_progress, {
                'loss': train_loss / (batch_progress.n + 1e-5),
                'acc': 100. * correct / total
            })

        batch_progress.close()
        epoch_loss = train_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def _validate(self, epoch, progress):
        """验证阶段"""
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0

        batch_progress = progress.batch_progress('val', epoch)
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                batch_progress.update(1)
                progress.update_metrics(batch_progress, {
                    'loss': val_loss / (batch_progress.n + 1e-5),
                    'acc': 100. * correct / total
                })

        batch_progress.close()
        epoch_loss = val_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def train(self):
        """完整的训练流程"""
        total_epochs = self.config['training']['epochs']
        progress = TrainingProgress(
            total_epochs=total_epochs,
            train_loader_len=len(self.train_loader),
            val_loader_len=len(self.val_loader),
            metrics=['loss', 'acc']
        )

        self.logger.info("🚀 开始训练模型...")
        self.logger.info(f"设备类型: {self.device}")
        self.logger.info(f"训练参数: {self.config['training']}")

        for epoch in range(1, total_epochs + 1):
            epoch_progress = progress.epoch_progress(epoch)

            # 训练阶段
            train_loss, train_acc = self._train_epoch(epoch, progress)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # 验证阶段
            val_loss, val_acc = self._validate(epoch, progress)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 学习率调整
            self.scheduler.step(val_acc)

            # 保存最新模型
            self._save_model(epoch)

            # 保存最佳模型
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self._save_model(epoch, is_best=True)
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            # 显示最近5个epoch的指标
            recent_epochs = self.history['val_acc'][-5:]
            epoch_progress.set_postfix({
                'best': f"{self.best_acc:.2f}%",
                'recent': f"{min(recent_epochs):.2f}~{max(recent_epochs):.2f}%"
            })
            epoch_progress.close()

            # 早停检测
            if self.config['training']['early_stopping']['enable'] and \
                    self.early_stop_counter >= self.config['training']['early_stopping']['patience']:
                self.logger.warning(f"⚠️ 早停触发：连续{self.early_stop_counter}个epoch未提升验证准确率")
                break

        self.logger.info("✅ 训练完成")


def main():
    config = Loader.load_config()
    trainer = MNISTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
