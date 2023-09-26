import logging

import torch
from torch import nn

from TranscriptionDataset.TranscriptionDataset import getDataPipe
from model import RWKV_Init, Block, L2Wrap, GPTConfig

logger = logging.getLogger(__name__)


class RWKVEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.step = 0
        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.Sequential(*[Block(config, i)
                                      for i in range(config.n_layer)])

        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # if RWKV_HEAD_QK_DIM > 0:
        #     self.head_q = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)
        #     self.head_q.scale_init = 0
        #     self.head_k = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)
        #     self.head_k.scale_init = 0.1
        #     self.register_buffer("copy_mask", torch.tril(
        #         torch.ones(config.ctx_len, config.ctx_len)))

        self.ctx_len = config.ctx_len

        RWKV_Init(self, config)

        logger.info("number of parameters: %e", sum(p.numel()
                                                    for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def configure_optimizers(self, train_config):
        no_decay = set()

        for mn, m in self.named_modules():  # here we disable weight_decay
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        # try:
        #     optimizer = FusedAdam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # except:
        print('\n\nDeepSpeed not found. Using torch optimizer instead (probably slower)\n\n')
        optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas,
                                     eps=train_config.eps)

        return optimizer

    def forward(self, embeds, targets=None):
        # embeds = embeds.to(self.emb.weight.device)
        embeds = embeds.to(self.device)

        self.step += 1
        B, T, E = embeds.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        x = self.blocks(embeds)
        x = self.ln_out(x)

        # if RWKV_HEAD_QK_DIM > 0:
        #     q = self.head_q(x)[:, :T, :]
        #     k = self.head_k(x)[:, :T, :]
        #     c = (q @ k.transpose(-2, -1)) * (1.0 / RWKV_HEAD_QK_DIM)
        #     c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)
        #
        #     if '32' in os.environ['RWKV_FLOAT_MODE']:
        #         c = c @ F.one_hot(idx, num_classes=self.config.vocab_size)
        #     elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
        #         c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).half()
        #     elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
        #         c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).bfloat16()
        #
        #     x = self.head(x) + c
        # else:

        # x = self.head(x)

        # loss = None
        # if targets is not None:
        #     loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.to(x.device).view(-1))

        # return L2Wrap.apply(loss, x)
        return x

    def get_ctx_len(self):
        return self.ctx_len


def test():
    datasetLocation = "../Trainsets/S_Tier_1695428558_mTokens1000_mNoS60.hdf5"
    dataset, pipe = getDataPipe(datasetLocation, 2)
    itx = iter(pipe)
    x = next(itx)
    all_mels, src_mask, src_pad_mask, all_tokens, tgt_mask, tgt_pad_mask, tokens_out = x
    config = GPTConfig(
        dataset.getVocabSize(),
        512,
        n_embd=512,
        n_layer=3,
    )
    model = RWKVEncoder(config)

    pass


if __name__ == '__main__':
    test()
