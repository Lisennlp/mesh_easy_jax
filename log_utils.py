import datetime
import logging
import os
import jax

class CustomLogger(logging.Logger):
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        if extra is None:
            extra = {}
        extra['host_id'] = jax.process_index()
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)

def setup_logger(host_id):
    logger = CustomLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    if not os.path.exists('logs'):
        os.mkdir('logs')
    log_filename = f"logs/log_{current_time}_{host_id}.txt"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s-%(filename)s-%(host_id)s] - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # 注册一个清理函数，在关闭日志处理器时上传日志文件
    def cleanup():
        file_handler.close()
        command = f'gsutil cp {log_filename} gs://jax_llm_logs/'
        response = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    # 在 Python 解释器关闭时自动执行清理函数
    import atexit
    atexit.register(cleanup)
    return logger

# 使用示例
logger = setup_logger(jax.process_index())