import torch
from transformers import BertTokenizer, BertModel, PreTrainedTokenizer, PreTrainedModel
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


class BaseCHEmbedding(BaseModel, ABC):
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def __call__(self, sentences):
        """
        获取向量
        sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
        """
        sentences = list(sentences)
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        return sentence_embeddings

    @classmethod
    @abstractmethod
    def init_model(self, model_path):
        pass


class Text2VecBase(BaseCHEmbedding):

    @classmethod
    def init_model(cls, model_path):
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertModel.from_pretrained(model_path)
        model.eval()
        return cls(tokenizer=tokenizer, model=model)
