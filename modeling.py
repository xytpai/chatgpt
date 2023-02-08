import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    hidden_size: int = 256
    num_attention_heads: int = 16
    vocab_size: int = 30522
    dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    num_layers: int = 12
    ignore_index: int = -1


class CausalAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.attention_dropout = nn.Dropout(config.dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden_dropout = nn.Dropout(config.dropout_prob)
        self.hidden_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.attention_out_layer = nn.Sequential(
                nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=False), 
                nn.GELU(),
                nn.Linear(2 * config.hidden_size, config.hidden_size), 
                nn.Dropout(config.dropout_prob))
        self.register_buffer("bias", torch.tril(
            torch.ones(config.max_position_embeddings, config.max_position_embeddings)
                ).view(1, 1, config.max_position_embeddings, config.max_position_embeddings))

    def forward(self, x):
        batch_size, seq_length, hidden_size = x.size()
        q, k, v = self.qkv(self.input_layernorm(x)).split(hidden_size, dim=2)
        q = q.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2) # -> (b, nh, t, hs)
        k = k.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2) # -> (b, nh, t, hs)
        v = v.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2) # -> (b, nh, t, hs)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.attention_head_size) # -> (b, nh, t, t)
        attention_scores = attention_scores.masked_fill(self.bias[:,:,:seq_length,:seq_length] == 0, float('-inf'))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs) # -> (b, nh, t, t)
        output = torch.matmul(attention_probs, v) # -> (b, nh, t, hs)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        output = x + self.hidden_dropout(self.dense(output))
        output = output + self.attention_out_layer(self.hidden_layernorm(output))
        return output


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embeddings_dropout = nn.Dropout(config.dropout_prob)
        self.blocks = nn.ModuleList([CausalAttentionBlock(config) for _ in range(config.num_layers)])
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.word_embeddings.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('attention_out_layer.2.weight') or pn.endswith('dense.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None):
        device = input_ids.device
        _, seq_length = input_ids.size()
        assert seq_length <= self.config.max_position_embeddings
        pos = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0) # (1, t)
        word_embeddings = self.word_embeddings(input_ids) # (b, t, hidden_size)
        position_embeddings = self.position_embeddings(pos) # (1, t, n_embd)
        embeddings = self.embeddings_dropout(word_embeddings + position_embeddings)
        for block in self.blocks:
            embeddings = block(embeddings)
        embeddings = self.layernorm(embeddings)
        if targets is not None:
            output =  self.lm_head(embeddings) # b, t, vocab_size
            loss = F.cross_entropy(output.view(-1, output.size(-1)), 
                targets.view(-1), ignore_index=self.config.ignore_index)
        else:
            output = self.lm_head(embeddings[:, [-1], :]) # b, 1, vocab_size
            loss = None
        return output, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at max_position_embeddings
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_position_embeddings \
                else input_ids[:, -self.config.max_position_embeddings:]
            # forward the model to get the logits for the index in the sequence
            output, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            output = output[:, -1, :] / temperature # b, vocab_size
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(output, min(top_k, output.size(-1)))
                output[output < v[:, [-1]]] = -float('Inf')
            idx_next = torch.multinomial(F.softmax(output, dim=-1), num_samples=1) # b
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == '__main__':
    config = GPTConfig()
    print(config)
    model = GPT(config)
    # print(model)
    fake_input = torch.rand(2, 10) * 100  # batch_size, seq_length
    fake_input = fake_input.long()
    pred, _ = model(fake_input)
    print(pred)
    masked_lm_labels = torch.randn(2, 10) * 100
    masked_lm_labels = masked_lm_labels.long()
    masked_lm_labels[masked_lm_labels < 50] = -1
    pred, loss = model(fake_input, masked_lm_labels)
    print(loss)
    loss.backward()
