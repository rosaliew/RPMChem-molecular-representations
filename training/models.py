import glob
import inspect
import json
import math
import os
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten





@dataclass 
class LlamaArgs: # following some aspects of this https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/llama.py
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim = None
    max_position_embeddings = None
    num_key_value_heads = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    tie_word_embeddings: bool = True

    
    @classmethod
    def from_dict(cls, params): # Reference to https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/base.py#L12 (I used this directly and copied it so credits go to them)
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

# look https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/tuner/lora.py#L11
class LoRAInfusedLinear(nn.Module):
        def __init__(
        self,
        input_dims,
        output_dims,
        r=8,
        dropout=0.0,
        scale=20.0,
        bias=False):
            
        super().__init__() 
        # this is finally very pytorch-ish
        self.linear = nn.Linear(input_dims, output_dims, bias=bias) 
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale

        # best practices from https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/tuner/lora.py#L13
        init_scale = 1.0 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(input_dims, r), # so d_model x rank, then rank x d_model (or output_dims more generally as it depends on the layer it is applied to)
        )
        self.lora_b = mx.zeros((r, output_dims))

    def __call__(self, x): # from https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/tuner/lora.py#L13
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)

    @staticmethod
    def from_base(linear, rank=8, dropout=0.0, scale=20.0): # converts linear to Lora linear based form (called it LoRAInfusedLinear) for now
        dim_out, dim_in= linear.weight.shape

        if isinstance(linear, nn.QuantizedLinear):
            dim_in = dim_in * 32 // linear.bits # calculate the true in_dims by considering how it was quantized. If linear.bits is 4 then its 4 bit quantization

        layer = LoRAInfusedLinear(
            input_dims=dim_in,
            output_dims=dim_out,
            r=rank,
            dropout=dropout,
            scale=scale,
        )

        layer.linear = linear # this just replaces the original linear layer with a lora linear layer
        return layer


# HEAVILY INSPIRED FROM https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/models/llama.py
# HEAVILY INSPIRED FROM https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/models/llama.py
# HEAVILY INSPIRED FROM https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/models/llama.py
# HEAVILY INSPIRED FROM https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/models/llama.py

class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        d_model = args.hidden_size

        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or (args.hidden_size // args.num_attention_heads)

        self.scale = self.head_dim**-0.5 # from the original attention is all you need paper, they scale the dot product by 1/sqrt(d_k).

        self.q_proj = nn.Linear(d_model, self.n_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d_model, bias=args.attention_bias)

        # simplier implementation then what they have in the paper but it seems to work fine.
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

    def __call__(self, x, mask=None):
        batch_size, max_seq_len, _ = x.shape

        q = self.q_proj(x).reshape(batch_size, max_seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, max_seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, max_seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, max_seq_len, -1)
        return self.o_proj(out)


# inspired from https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/models/llama.py#L13
# inspired from https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/models/llama.py#L13
# inspired from https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/models/llama.py#L13

class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args.hidden_size
        hidden = args.intermediate_size

        self.gate_proj = nn.Linear(dim, hidden, bias=args.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden, bias=args.mlp_bias)
        self.down_proj = nn.Linear(hidden, dim, bias=args.mlp_bias)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x)) # same as llama SwishGelu thing defined by apples mlx (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/activations.py#L10)

class TransformerBlock(nn.Module): # inspired from (but modified) https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/models/llama.py#L13
     def __init__(self, args):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None):
        h = x + self.self_attn(self.input_layernorm(x), mask) # residual connections for h and residual connections for out
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

# heavily inspired from https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/models/llama.py#L13
class LlamaModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, input_ids):
        h = self.embed_tokens(input_ids)
        mask = "causal" if h.shape[1] > 1 else None # apply causal mask if sequence length is more than 1 (if its just 1 token then we dont need to apply the mask because there is nothing to mask)
        for layer in self.layers:
            h = layer(h, mask)
        return self.norm(h)

#https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/models/llama.py
class LlamaForCausalLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property # this is from the original source, I would have just not put this decorator here and instead just called model_obj.layers (however, this is probably better practice)
    def layers(self):
        return self.model.layers

    def __call__(self, input_ids):
        out = self.model(input_ids)
        if self.args.tie_word_embeddings: 
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights): # also part of original implementation - is used to remove unused stuff from the buffer
        weights = {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights



def load_config(model_dir):
    with open(f"{model_dir}/config.json", "r", encoding="utf-8") as f:
        return json.load(f)


# see https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/utils.py for class 
def load_pretrained_model(model_dir):
    config = load_config(model_dir)
    args = LlamaArgs.from_dict(config)

    model = LlamaForCausalLM(args)

    weight_files = sorted(glob.glob(f"{model_dir}/*.safetensors")) # look for anything with the .safetensors filename

    weights = {}
    for wf in weight_files: # sometimes weights are split across many files so this handles it - for LLama8b we dont need it tho
        weights.update(mx.load(wf))


    weights = model.sanitize(weights)

    quantization = config.get("quantization", None) # get quantization details from the model dir config thing
    if quantization is None:
        quantization = config.get("quantization_config", None)

    if quantization is not None:
        # see https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/utils.py for class predicate. this is for converting a 4bit loaded in model into its actual 4 bits (its stored full precsion with scale params)
        def class_predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            return f"{path}.scales" in weights

        nn.quantize(model,group_size=quantization["group_size"],bits=quantization["bits"],mode=quantization.get("mode", "affine") class_predicate=class_predicate)

    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters()) # mlx is lazy eval, so this forces it to work
    return model, config


def linear_to_lora_layers( # adapted and simplified from https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/utils.py
    model,
    num_layers,
    rank,
    scale,
    dropout=0.0,
):
    keys = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }

    def to_lora(layer):
        return LoRAInfusedLinear.from_base(layer, rank=rank, scale=scale, dropout=dropout)


    # from https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/utils.py
    # from https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/utils.py
    # from https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/utils.py
    for block in model.layers[-max(num_layers, 0) :]:
        updates = [(k, to_lora(m)) for k, m in block.named_modules() if k in keys]
        if updates:
            block.update_modules(tree_unflatten(updates))



def save_lora_adapters(model, path): # Used a similar method of this https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py
    os.makedirs(os.path.dirname(path), exist_ok=True)
    adapters = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(path, adapters)


def causal_lm_loss(logits, labels, ignore_index=-100):
    shift_logits = logits[:, :-1, :] # go to very last token in sequence
    shift_labels = labels[:, 1:] # ground truth is shifted by one as we want to predict the next token

    mask = shift_labels != ignore_index # we want to make a mask where we ignore the tokens that are -100 (padding tokens)
    safe_targets = mx.where(mask, shift_labels, 0) # replace the -100 tokens with 0 for now. This is because my pad token -100 is nonsensical and cannot be indexed for the vocab vector. Thus, we conver it to any valid token index (like 0) and compute loss (we ignore the loss at this token anyways) - we mult by mask later

    ce = nn.losses.cross_entropy(shift_logits, safe_targets)
    ce = ce * mask.astype(ce.dtype)

    # from https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py
    ntoks = mx.maximum(mask.sum(), 1)
    return ce.sum() / ntoks
