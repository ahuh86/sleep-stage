# ============================================================
# logger.py — 日志模块
#
# 【这个文件是干什么的？】
# 控制台和文件双输出的日志系统。
# 训练过程中的所有信息（loss、accuracy、配置等）都会：
#   1. 显示在控制台（带颜色高亮）
#   2. 写入日志文件（带时间戳，便于后续查看）
# ============================================================

import os
import logging as py_logging


# 日志级别映射表
# DEBUG: 最详细的调试信息
# INFO:  一般信息（训练进度、指标等）
# WARNING: 警告信息
# ERROR: 错误信息
_log_level = {
    None: py_logging.NOTSET,
    'debug': py_logging.DEBUG,
    'info': py_logging.INFO,
    'warning': py_logging.WARNING,
    'error': py_logging.ERROR,
    'critical': py_logging.CRITICAL
}


def get_logger(log_file_path=None, name='default_log', level=None):
    """获取一个配置好的logger实例。

    这个logger会同时输出到：
      - 控制台：带颜色格式，适合实时查看
      - 文件：带时间戳，适合存档分析

    参数:
        log_file_path: str or None, 日志文件的保存路径
                        如果为None，只输出到控制台
        name:          str, logger名称，用于区分不同的logger实例
        level:         str or None, 日志级别（'debug', 'info', 'warning', 'error', 'critical'）

    返回:
        logging.Logger 实例
    """
    # 确保日志目录存在
    directory = os.path.dirname(log_file_path)
    if os.path.isdir(directory) and not os.path.exists(directory):
        os.makedirs(directory)

    root_logger = py_logging.getLogger(name)
    handlers = root_logger.handlers

    def _check_file_handler(logger, filepath):
        for handler in logger.handlers:
            if isinstance(handler, py_logging.FileHandler):
                return handler.baseFilename == os.path.abspath(filepath)
        return False

    # 添加文件处理器（如果尚未添加）
    if (log_file_path is not None and not
            _check_file_handler(root_logger, log_file_path)):
        log_formatter = py_logging.Formatter(
            '%(asctime)s [%(levelname)-5.5s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        file_handler = py_logging.FileHandler(log_file_path)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    # 添加控制台处理器（如果尚未添加，带ANSI颜色）
    if any([type(h) == py_logging.StreamHandler for h in handlers]):
        return root_logger
    level_format = '\x1b[36m[%(levelname)-5.5s]\x1b[0m'
    log_formatter = py_logging.Formatter(f'{level_format} %(message)s')
    console_handler = py_logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(_log_level[level])
    return root_logger


if __name__ == '__main__':
    logger = get_logger('test.log', name='test', level='info')
    logger.info('Test')
    logger.info('Test2')
    logger.info('Test3')
