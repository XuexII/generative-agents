import openai
import secrets
from pydantic import BaseModel
from typing import Dict, Set, Tuple, Optional
from pydantic import BaseModel, Field
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from queue import Queue
import tiktoken
import logging
import openai
import openai.error
import time



def encrypt_string(plaintext, key):
    cipher = AES.new(key, AES.MODE_ECB)
    ciphertext = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
    return ciphertext


def decrypt_string(ciphertext, key):
    cipher = AES.new(key, AES.MODE_ECB)
    decrypted = cipher.decrypt(ciphertext)
    plaintext = unpad(decrypted, AES.block_size)
    return plaintext.decode()

MAX_TOKENS = 30000


class RichKey(BaseModel):
    rich_key: str = Field(alias="key")
    encrypt_key: str
    encrypt_text: str
    sold: bool = Field(default=False)
    use_tokens: int = Field(default=0)
    avaliable: bool = Field(default=True)
    # TODO 每月更新
    update_time: str

    def decrypt(self):
        return decrypt_string(self.encrypt_text, self.encrypt_key)

    def legal(self):
        return self.avaliable

    def update_tokens(self, num_tokens):
        self.use_tokens += num_tokens
        if self.use_tokens > MAX_TOKENS:
            raise RuntimeError(f"超过使用限制")


class RichOpenai:
    rich_keys: Dict[str, RichKey] = {}
    no_sold: Queue = Queue()

    def __init__(self, key_path):
        self.wrap_keys(key_path)
        self.args = {
            "model": "gpt-3.5-turbo",  # 对话模型的名称
            "temperature": 1.0,  # 值在[0,1]之间，越大表示回复越具有不确定性
            "max_tokens": 4096,  # 回复最大的字符数
            "top_p": 1,
            "frequency_penalty": 0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            "presence_penalty": 0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            "request_timeout": None,  # 请求超时时间，openai接口默认设置为600，对于难问题一般需要较长时间
            "timeout": None,  # 重试超时时间，在这个时间内，将会自动重试
        }

    def load_keys(self, path):
        keys = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                key, up_date = line.split("\t")
                keys.append({"key": key, "up_date": up_date})
        return keys

    def wrap_keys(self, path):
        keys = self.load_keys(path)
        for itm in keys:
            key = itm["key"]
            update_time = itm["up_date"]
            rich_key = secrets.token_hex(32)
            encrypt_key = get_random_bytes(32)
            encrypt_text = encrypt_string(key, encrypt_key)
            self.rich_keys[rich_key] = RichKey(rich_key=rich_key,
                                               encrypt_key=encrypt_key,
                                               encrypt_text=encrypt_text,
                                               update_time=update_time)
            self.no_sold.put(rich_key)

    def sell(self):
        if not self.no_sold.empty():
            return self.no_sold.get()
        return None

    def valid(self, rich_key):
        if rich_key not in self.rich_keys:
            raise RuntimeError(f"无效的key")
        rich_obj: RichKey = self.rich_keys[rich_key]
        if not rich_obj.legal():
            raise RuntimeError(f"超过使用限制")

    # 请求openai
    def reply_text(self, rich_key, messages, **kwargs):
        for k, v in kwargs:
            if k in self.args:
                self.args[k] = v
        try:
            self.valid(rich_key)
            model = self.args["model"]
            rich_obj: RichKey = self.rich_keys[rich_key]
            num_tokens = num_tokens_from_messages(messages, model)
            rich_obj.update_tokens(num_tokens)

            response = openai.ChatCompletion.create(api_key=rich_obj.decrypt(), messages=messages, **kwargs)
            return response
        except Exception as e:
            return repr(e)


def num_tokens_from_messages(messages, model):
    """Returns the number of tokens used by a list of messages."""

    if model in ["gpt-3.5-turbo-0301", "gpt-35-turbo"]:
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo")
    elif model in ["gpt-4-0314", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613", "gpt-35-turbo-16k"]:
        return num_tokens_from_messages(messages, model="gpt-4")

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logging.warning("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        logging.warning(f"num_tokens_from_messages() is not implemented for model {model}. Returning num tokens assuming gpt-3.5-turbo.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens














