import logging
import time
import traceback
from threading import Thread, Lock

import torch


class EmptyGPUCacheTimer(Thread):

    def __init__(self, lock: Lock):
        super().__init__(daemon=True)
        self.lock = lock
        self.interval = 10  # 清理间隔
        self.is_running = True

    def get_gpu_memory(self):
        s = 0
        s += torch.cuda.memory_allocated()
        s += torch.cuda.memory_reserved()
        return s / 1024 / 1024

    def run(self):
        logging.info(f'定时清缓存线程已启动，每{self.interval}秒会清理GPU缓存，等待GPU：{self.wait_lock}。')
        while self.is_running:
            time.sleep(self.interval)
            with self.lock:
                try:
                    a = self.get_gpu_memory()
                    torch.cuda.empty_cache()
                    b = self.get_gpu_memory()
                    if a != b:
                        logging.debug(f'清理了：{a - b:.2f}MB显存。')
                except:
                    traceback.print_exc()
                    pass
        logging.info(f'定时清缓存线程已退出。')
