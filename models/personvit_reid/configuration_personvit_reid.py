from transformers import PretrainedConfig


class PersonViTReIDConfig(PretrainedConfig):
    model_type = "personvit-reid"

    def __init__(
        self,
        image_size: list[int] | tuple[int, int] = (256, 128),
        patch_size: int = 16,
        num_channels: int = 3,
        hidden_size: int = 384,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 6,
        intermediate_size: int = 1536,
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        qkv_bias: bool = True,
        embedding_dim: int = 384,
        normalize_embeddings: bool = True,
        reid_target_size: list[int] | tuple[int, int] = (128, 256),
        reid_aspect_ratio: float = 0.5,
        backbone_name: str = "personvit-msmt17-vit-s",
        pooling: str = "cls",
        backbone_num_classes: int = 1000,
        classifier_out_features: int = 1041,
        use_bnneck: bool = True,
        **kwargs,
    ):
        if len(image_size) != 2:
            raise ValueError("image_size must contain [height, width]")
        if len(reid_target_size) != 2:
            raise ValueError("reid_target_size must contain [width, height]")
        self.image_size = [int(image_size[0]), int(image_size[1])]
        self.patch_size = int(patch_size)
        self.num_channels = int(num_channels)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size)
        self.hidden_dropout_prob = float(hidden_dropout_prob)
        self.attention_probs_dropout_prob = float(attention_probs_dropout_prob)
        self.layer_norm_eps = float(layer_norm_eps)
        self.initializer_range = float(initializer_range)
        self.qkv_bias = bool(qkv_bias)
        self.embedding_dim = int(embedding_dim)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.reid_target_size = [int(reid_target_size[0]), int(reid_target_size[1])]
        self.reid_aspect_ratio = float(reid_aspect_ratio)
        self.backbone_name = backbone_name
        self.pooling = pooling
        self.backbone_num_classes = int(backbone_num_classes)
        self.classifier_out_features = int(classifier_out_features)
        self.use_bnneck = bool(use_bnneck)
        super().__init__(**kwargs)
