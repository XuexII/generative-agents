import logging
import os


class Logger:

    def __init__(self,
                 path,
                 clevel=logging.DEBUG,
                 flevel=logging.DEBUG,
                 fmt='[%(asctime)s] [%(levelname)s] %(message)s'):
        self.logger = logging.getLogger(path)

        self.logger.setLevel(logging.DEBUG)

        fmt = logging.Formatter(fmt, '%Y-%m-%d %H:%M:%S')

        # 设置CMD日志

        sh = logging.StreamHandler()

        sh.setFormatter(fmt)

        sh.setLevel(clevel)

        # 设置文件日志

        fh = logging.FileHandler(path, encoding="utf-8")

        fh.setFormatter(fmt)

        fh.setLevel(flevel)

        self.logger.addHandler(sh)

        self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def war(self, message):
        self.logger.warn(message)

    def error(self, message):
        self.logger.error(message)

    def cri(self, message):
        self.logger.critical(message)