import logging
import os
from multiprocessing.util import DEFAULT_LOGGING_FORMAT
from  datetime import datetime
from utils.path_tool import get_abs_path

#日志保存的根目录
LOG_ROOT = get_abs_path("logs")
# 创建日志目录
# 如果目录已存在，则不会抛出异常，因为设置了exist_ok=True参数
os.makedirs(LOG_ROOT, exist_ok=True)


# 设置默认日志格式的常量
DEFAULT_LOG_FORMAT = logging.Formatter(
    # 日志格式字符串，包含以下元素：
    # %(asctime)s - 日志事件发生的时间
    # %(name)s - 日志记录器的名称
    # %(levelname)s - 日志级别
    # %(filename)s - 发出日志请求的文件名
    # %(lineno)d - 发出日志请求的代码行号
    # %(message)s - 日志消息内容
    "%(asctime)s - %(name)s %(levelname)s - %(filename)s:%(lineno)d %(message)s",
)
def get_logger(
        name:str ="agent",
        console_level:int = logging.INFO,
        file_level:int = logging.DEBUG,
        log_file:str = None,
)-> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
    filr_handler = logging.FileHandler(log_file,encoding="utf-8")
    filr_handler.setLevel(file_level)
    filr_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(filr_handler)
    return logger

logger = get_logger()
if __name__ == "__main__":
    logger.info("this is a infotest")
    logger.error("this is a errortest")
    logger.debug("this is a debugtest")