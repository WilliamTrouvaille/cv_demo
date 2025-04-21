#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/03/21 20:30
@version : 1.0.0
@author  : William_Trouvaille
@function: 日志工具类
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.utils import Loader


class LoggerHandler:
    """
    单例模式的日志封装类
    功能：支持控制台+文件双输出、自动创建日志目录、动态时间戳命名
    用法：
    1. 实例化调用：logger = LoggerHandler(); logger.info("message")
    2. 类方法调用：LoggerHandler.info("message")
    """
    _instance = None

    def __new__(cls, level=logging.DEBUG,
                fmt='[%(asctime)s] [%(levelname)s] %(message)s'):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            # 初始化日志器
            cls._instance.logger = logging.getLogger("LoggerHandler")
            cls._instance.logger.setLevel(level)
            root = Loader.get_root_dir()

            # 创建日志目录
            log_dir = Path(root) / "logs"
            log_dir.mkdir(exist_ok=True)

            # 定义格式器
            formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

            # 控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)

            # 文件处理器（按日期命名）
            log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)

            # 添加处理器
            cls._instance.logger.addHandler(console_handler)
            cls._instance.logger.addHandler(file_handler)

        return cls._instance

    @classmethod
    def debug(cls, msg):
        """类方法：记录调试信息"""
        cls._instance.logger.debug(msg)

    @classmethod
    def info(cls, msg):
        """类方法：记录一般信息"""
        cls._instance.logger.info(msg)

    @classmethod
    def warning(cls, msg):
        """类方法：记录警告信息"""
        cls._instance.logger.warning(msg)

    @classmethod
    def error(cls, msg):
        """类方法：记录错误信息"""
        cls._instance.logger.error(msg)

    @classmethod
    def critical(cls, msg):
        """类方法：记录严重错误信息"""
        cls._instance.logger.critical(msg)
