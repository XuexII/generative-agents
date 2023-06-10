import logging
from abc import ABC

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
from utils.utils_cuda import EmptyGPUCacheTimer
from threading import Lock

model_path = "/home/maiyuan/xuyongzhao/pretrainedmodels/THUDM-chatglm-6b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()

print()

model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)


class ChatGLM(ABC):
    history: List = []

    def __init__(self):
        logging.info(f"开始加载GLM模型".center(66, "="))
        model_path = "/home/maiyuan/xuyongzhao/pretrainedmodels/THUDM-chatglm-6b-int4"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        use_cuda = torch.cuda.is_available()
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half()
        self.lock = Lock()
        if use_cuda:
            self.model = self.model.cuda()
            egct = EmptyGPUCacheTimer(self.lock)
            egct.start()
            self.egct = egct
        self.model = self.model.eval()
        logging.info(f"加载GLM模型成功".center(66, "="))
        self.args = {"max_length": 2048,  # 0-4096
                     "top_p": 0.7,  # 0-1
                     "temperature": 0.95  # 温度
                     }

    def processing(self, text, save_history=False):
        response, history = self.model.chat(self.tokenizer, text, history=self.history, **self.args)
        if save_history:
            self.history = history
        return response

    def __del__(self):
        if hasattr(self, 'egct') and self.egct is not None:
            self.egct.is_running = False


chat_glm = ChatGLM()
