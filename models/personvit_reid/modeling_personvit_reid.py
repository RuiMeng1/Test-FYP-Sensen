from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_personvit_reid import PersonViTReIDConfig


@dataclass
class PersonViTReIDOutput(ModelOutput):
    embeddings: torch.FloatTensor | None = None
    global_features: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size: int,
        in_chans: int,
        embed_dim: int,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (
            img_size[0] // patch_size,
            img_size[1] // patch_size,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, config: PersonViTReIDConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(
            config.hidden_size,
            config.hidden_size * 3,
            bias=config.qkv_bias,
        )
        self.attn_drop = nn.Dropout(config.attention_probs_dropout_prob)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_drop = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)
        x = (attention @ value).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, config: PersonViTReIDConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.drop = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, config: PersonViTReIDConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Mlp(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PersonViTBackbone(nn.Module):
    def __init__(self, config: PersonViTReIDConfig):
        super().__init__()
        image_height, image_width = config.image_size
        if image_height % config.patch_size != 0 or image_width % config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_embed = PatchEmbed(
            img_size=(image_height, image_width),
            patch_size=config.patch_size,
            in_chans=config.num_channels,
            embed_dim=config.hidden_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, config.hidden_size)
        )
        self.pos_drop = nn.Dropout(config.hidden_dropout_prob)
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc = nn.Linear(config.hidden_size, config.backbone_num_classes)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0], x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls_token, hidden_states = self.forward_features(x)
        logits = self.fc(cls_token)
        return logits, hidden_states


class PersonViTReIDModel(PreTrainedModel):
    config_class = PersonViTReIDConfig
    base_model_prefix = "personvit_reid"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def __init__(self, config: PersonViTReIDConfig):
        super().__init__(config)
        self.base = PersonViTBackbone(config)
        self.bottleneck = nn.BatchNorm1d(config.hidden_size)
        self.bottleneck.bias.requires_grad_(False)
        self.projection = (
            nn.Identity()
            if config.embedding_dim == config.hidden_size
            else nn.Linear(config.hidden_size, config.embedding_dim, bias=False)
        )
        self.classifier = nn.Linear(
            config.hidden_size,
            config.classifier_out_features,
            bias=False,
        )
        self.post_init()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _check_input_size(self, pixel_values: torch.Tensor):
        expected_height, expected_width = self.config.image_size
        if pixel_values.shape[-2:] != (expected_height, expected_width):
            raise ValueError(
                "Expected pixel_values shape (*, 3, "
                f"{expected_height}, {expected_width}), "
                f"received {tuple(pixel_values.shape)}"
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool | None = None,
        **_: dict,
    ):
        self._check_input_size(pixel_values)
        global_features, hidden_states = self.base.forward_features(pixel_values)
        if self.config.use_bnneck:
            embeddings = self.bottleneck(global_features)
        else:
            embeddings = global_features
        embeddings = self.projection(embeddings)
        if self.config.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        if return_dict:
            return PersonViTReIDOutput(
                embeddings=embeddings,
                global_features=global_features,
                last_hidden_state=hidden_states,
            )
        return embeddings, hidden_states
