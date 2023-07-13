import datetime
import logging
import os
import jax
import subprocess
import smart_open


# 重写logger的log函数
class CustomLogger(logging.Logger):
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        if extra is None:
            extra = {}
        extra['host_id'] = jax.process_index()
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)

# 重写FileHander的open函数
class CustomHandler(logging.FileHandler):
    def _open(self):
        if 'gs:' in self.baseFilename:
            filename = self.baseFilename.split('gs:')[-1]
            # filename: /jax_llm_logs/....
            return smart_open.open(f'gs:/{filename}', 'w')
        else:
            return open(self.baseFilename, self.mode, encoding=self.encoding)

def setup_logger(host_id):
    logger = CustomLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    # 只把主节点的日志写入到bucket
    if host_id == 0:
        log_filename = f"gs://jax_llm_logs/train_log_{current_time}_{host_id}.txt"
    else:
        log_filename = f"log_{current_time}_{host_id}.txt"
    file_handler = CustomHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s-%(filename)s-%(host_id)s] - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger(jax.process_index())