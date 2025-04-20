from pathlib import Path

import yaml


class Loader:
    @staticmethod
    def get_root_dir() -> Path:
        """获取项目根目录（自动定位）"""
        # 通过当前文件路径回溯定位项目根目录（src/utils -> ../../）
        current_dir = Path(__file__).parent
        return current_dir.parent.parent

    @staticmethod
    def get_path(relative_path: str) -> Path:
        """
        输入相对于项目根目录的地址（文件或目录），返回Path对象
        """
        if relative_path.startswith('/'):
            relative_path = relative_path[1:]
        return Loader.get_root_dir().joinpath(relative_path)

    @classmethod
    def load_config(cls, config_name='config.yaml') -> dict:
        """加载配置文件"""
        config_path = cls.get_path(r"config/" + config_name)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    @classmethod
    def build_path(cls, *path_parts) -> str:
        """构建基于项目根目录的绝对路径"""
        return str(cls.get_root_dir().joinpath(*path_parts))
