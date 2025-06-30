from transformers import PretrainedConfig


class KBLaMConfig(PretrainedConfig):
    """
    Configuration class for KBLaM BitNet models, extending HuggingFace PretrainedConfig.

    Args:
        base_model_name_or_path (str): Path or name of the base pretrained model. [default: ""]
        kb_layer_frequency (int): Frequency (in layers) to insert KB attention. [default: 3]
        kb_scale_factor (Optional[int]): Scaling factor for KB attention. [default: None]
        top_k_kb (int): Top-K KB entries to use for sparsification. [default: 100]
        dynamic_sparsify (bool): Whether to use dynamic KB sparsification. [default: False]
        sep_query_head (bool): Use separate query head for KB. [default: False]
        attn_implementation (str): Attention implementation type (e.g., "eager"). [default: "eager"]
        kb_length_scaling (bool): Enable KB length scaling in attention. [default: False]
        kb_max_train_triples (int): Max KB triples seen during training (for scaling). [default: 120000]
        ...kwargs: Additional config fields.
    """
    def __init__(
        self,
        base_model_name_or_path: str = "",
        kb_layer_frequency: int = 3,
        kb_scale_factor: int = None,
        top_k_kb: int = 100,
        dynamic_sparsify: bool = False,
        sep_query_head: bool = False,
        attn_implementation: str = "eager",
        kb_length_scaling: bool = False,
        kb_max_train_triples: int = 120000,
        enable_rotary: bool = True,
        enable_dropout: bool = True,
        enable_kb: bool = True,
        **kwargs,
    ):
        # enable_rotary: Toggle rotary embeddings (ablation). If False, disables rotary position embeddings everywhere.
        if not isinstance(enable_rotary, bool):
            raise TypeError("enable_rotary must be a boolean.")
        self.enable_rotary = enable_rotary

        # enable_dropout: Toggle all dropout (ablation). If False, disables all dropout in the model.
        if not isinstance(enable_dropout, bool):
            raise TypeError("enable_dropout must be a boolean.")
        self.enable_dropout = enable_dropout

        # enable_kb: Toggle KB integration (ablation). If False, disables all KB attention and related logic.
        if not isinstance(enable_kb, bool):
            raise TypeError("enable_kb must be a boolean.")
        self.enable_kb = enable_kb
        # --- Field docstrings and validation ---
        # base_model_name_or_path: Path or name of the base pretrained model.
        if not isinstance(base_model_name_or_path, str):
            raise TypeError("base_model_name_or_path must be a string.")
        self.base_model_name_or_path = base_model_name_or_path

        # kb_layer_frequency: Frequency (in layers) to insert KB attention.
        if not isinstance(kb_layer_frequency, int) or kb_layer_frequency < 1:
            raise ValueError("kb_layer_frequency must be a positive integer.")
        self.kb_layer_frequency = kb_layer_frequency

        # kb_scale_factor: Optional scaling factor for KB attention.
        if kb_scale_factor is not None and not isinstance(kb_scale_factor, int):
            raise TypeError("kb_scale_factor must be an int or None.")
        self.kb_scale_factor = kb_scale_factor

        # top_k_kb: Top-K KB entries to use for sparsification.
        if not isinstance(top_k_kb, int) or top_k_kb < 0:
            raise ValueError("top_k_kb must be a non-negative integer.")
        self.top_k_kb = top_k_kb

        # dynamic_sparsify: Whether to use dynamic KB sparsification.
        if not isinstance(dynamic_sparsify, bool):
            raise TypeError("dynamic_sparsify must be a boolean.")
        self.dynamic_sparsify = dynamic_sparsify

        # sep_query_head: Use separate query head for KB.
        if not isinstance(sep_query_head, bool):
            raise TypeError("sep_query_head must be a boolean.")
        self.sep_query_head = sep_query_head

        # attn_implementation: Attention implementation type.
        if not isinstance(attn_implementation, str):
            raise TypeError("attn_implementation must be a string.")
        self.attn_implementation = attn_implementation

        # kb_length_scaling: Enable KB length scaling in attention.
        if not isinstance(kb_length_scaling, bool):
            raise TypeError("kb_length_scaling must be a boolean.")
        self.kb_length_scaling = kb_length_scaling

        # kb_max_train_triples: Max KB triples seen during training (for scaling).
        if not isinstance(kb_max_train_triples, int) or kb_max_train_triples < 1:
            raise ValueError("kb_max_train_triples must be a positive integer.")
        self.kb_max_train_triples = kb_max_train_triples

        # --- Dropout and regularization fields ---
        # embd_pdrop: Dropout after embeddings
        embd_pdrop = kwargs.pop("embd_pdrop", 0.0)
        if not isinstance(embd_pdrop, (float, int)) or not (0.0 <= embd_pdrop <= 1.0):
            raise ValueError("embd_pdrop must be a float in [0.0, 1.0].")
        self.embd_pdrop = float(embd_pdrop)

        # attn_pdrop: Dropout after attention
        attn_pdrop = kwargs.pop("attn_pdrop", 0.0)
        if not isinstance(attn_pdrop, (float, int)) or not (0.0 <= attn_pdrop <= 1.0):
            raise ValueError("attn_pdrop must be a float in [0.0, 1.0].")
        self.attn_pdrop = float(attn_pdrop)

        # mlp_pdrop: Dropout after MLP
        mlp_pdrop = kwargs.pop("mlp_pdrop", 0.0)
        if not isinstance(mlp_pdrop, (float, int)) or not (0.0 <= mlp_pdrop <= 1.0):
            raise ValueError("mlp_pdrop must be a float in [0.0, 1.0].")
        self.mlp_pdrop = float(mlp_pdrop)

        # resid_pdrop: Dropout for residual connections
        resid_pdrop = kwargs.pop("resid_pdrop", 0.0)
        if not isinstance(resid_pdrop, (float, int)) or not (0.0 <= resid_pdrop <= 1.0):
            raise ValueError("resid_pdrop must be a float in [0.0, 1.0].")
        self.resid_pdrop = float(resid_pdrop)

        # classifier_dropout: Dropout in classifier heads
        classifier_dropout = kwargs.pop("classifier_dropout", 0.0)
        if not isinstance(classifier_dropout, (float, int)) or not (0.0 <= classifier_dropout <= 1.0):
            raise ValueError("classifier_dropout must be a float in [0.0, 1.0].")
        self.classifier_dropout = float(classifier_dropout)

        # activation_function: Activation for MLP
        activation_function = kwargs.pop("activation_function", "squared_relu")
        if activation_function not in ("squared_relu", "gelu", "swiglu"):
            raise ValueError("activation_function must be one of: 'squared_relu', 'gelu', 'swiglu'.")
        self.activation_function = activation_function
        # Call parent constructor for any additional fields
        super().__init__(**kwargs)
