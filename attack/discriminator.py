import torch
from torch import nn

from transformers import AutoTokenizer, AutoModel
from transformers import WEIGHTS_NAME, CONFIG_NAME

from src.logging_utils import info, finish


class Discriminator(nn.Module):
    def __init__(self, checkpoint: str = None) -> None:
        super().__init__()
        self.NET_CONFIG = 'net.pkl'
        self.hidden_size = 768
        self.dropout = nn.Dropout(0.2)
        self.model = None
        self.linear = nn.Linear(self.hidden_size * 2, 1)  # 长度为n的概率  需要尾部增加相同长度 0, 1 编码
        if checkpoint is None:
            info("load model from bert-base-uncased")
            self.model = AutoModel.from_pretrained('bert-base-uncased').to('cuda')
        else:
            info(f"load model from {checkpoint}")
            self.model = AutoModel.from_pretrained(checkpoint).to('cuda')
            state_dict = torch.load(f"{checkpoint}/{self.NET_CONFIG}")
            self.load_state_dict(state_dict)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        finish()

        self.to('cuda')

    def forward(self, state_infos, **kwargs):
        output = self.model(**kwargs)
        drop_output = self.dropout(output.last_hidden_state)  # [B, T, 768]
        # 增加 0, 1 值  state_info : [B, T, 1]   last_dim: 0/1
        state = state_infos.expand(state_infos.shape[0], state_infos.shape[1], 768)
        output = torch.cat((drop_output, state), dim=-1)  # 拼接
        logits = self.linear(output)
        return logits.squeeze(-1)  # [B, T]

    def saveModel(self, checkpoint: str = "checkpoint"):
        import os
        folder = os.path.exists(checkpoint)

        info(f"Model saved: {checkpoint}")
        if not folder:
            os.makedirs(checkpoint)
        info("PLM...")
        torch.save(self.model.state_dict(), f"{checkpoint}/{WEIGHTS_NAME}")
        self.model.config.to_json_file(f"{checkpoint}/{CONFIG_NAME}")
        info('\t保存PyTorch网络参数')
        torch.save(self.state_dict(), f"{checkpoint}/{self.NET_CONFIG}")
        f"{checkpoint}/{WEIGHTS_NAME}"
        finish()
