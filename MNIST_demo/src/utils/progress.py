import time
from typing import Dict

from tqdm.auto import tqdm


class TrainingProgress:
    """进度可视化组件"""

    def __init__(self,
                 total_epochs: int,
                 train_loader_len: int,
                 val_loader_len: int = 0,
                 metrics: list = ['loss', 'acc'],
                 colours: dict = None):
        """
        Args:
            total_epochs: 总训练轮次
            train_loader_len: 训练集批次数
            val_loader_len: 验证集批次数
            metrics: 需要展示的指标列表
            colours: 自定义颜色配置 {'train': 'green', 'val': 'yellow'}
        """
        self.total_epochs = total_epochs
        self.train_steps = train_loader_len
        self.val_steps = val_loader_len
        self.metrics = metrics
        self.colours = colours or {
            'train': '#00ff00',
            'val': '#ffff00',
            'test': '#00ffff'
        }

        # 训练计时
        self.epoch_start_time = None
        self.batch_start_time = None

    def epoch_progress(self, epoch: int) -> tqdm:
        """创建epoch级别进度条"""
        return tqdm(
            desc=f"Epoch {epoch}/{self.total_epochs}",
            bar_format="{l_bar}{bar:20}{r_bar}",
            colour=self.colours['train'],
            position=0,
            leave=True
        )

    def batch_progress(self,
                       mode: str = 'train',
                       current_epoch: int = None) -> tqdm:
        """创建batch级别进度条

        Args:
            mode: 训练模式 'train' or 'val'
            current_epoch: 当前epoch数（用于嵌套显示）
        """
        total_steps = self.train_steps if mode == 'train' else self.val_steps
        desc = f"├─ {mode.capitalize()}" if current_epoch else f"{mode.capitalize()}"
        # 颜色回退机制
        color = self.colours.get(
            mode,
            '#FFFFFF'  # 默认白色
        )

        return tqdm(
            total=total_steps,
            desc=desc,
            bar_format="  {l_bar}{bar:15}{r_bar}",
            colour=color,
            position=1,
            leave=False
        )

    def update_metrics(self,
                       pbar: tqdm,
                       metrics: Dict[str, float],
                       precision: int = 4) -> None:
        """动态更新指标显示"""
        formatted_metrics = {}
        for k, v in metrics.items():
            if k in self.metrics:
                if isinstance(v, float):
                    formatted_metrics[k] = f"{v:.{precision}f}"
                else:
                    formatted_metrics[k] = str(v)
        pbar.set_postfix(formatted_metrics)

    def time_since(self, start_time: float) -> str:
        """生成时间间隔描述"""
        seconds = time.time() - start_time
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h)}h{int(m)}m{int(s)}s"
