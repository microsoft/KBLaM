import json
import logging
import os
import pathlib
import numpy as np
import torch
import wandb

from accelerate import Accelerator
from rich.logging import RichHandler

from transformers import AutoTokenizer

from kblam.kb_encoder import KBEncoder
from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.models.bitnet_model import KBLaMBitNetForCausalLM

from train.args import parse_args
from train.config import get_prefix_str
from train.embeddings import _load_cached_embeddings
from train.params import _get_parameter_count
from train.retriever import KBRetriever
from train.trainer import Trainer
from train.ui import console


LOGFORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOGFORMAT_RICH = "%(message)s"

# setup logging
# Configure the root logger to WARNING
logging.basicConfig(
    level=logging.WARNING,  # Set the root logger to WARNING
    format=LOGFORMAT_RICH,
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)


def main():
    """The main function for training the knowledge base language model.

    This function orchestrates the entire training process. It initializes logging,
    parses command-line arguments, sets up the dataset, models, and tokenizer,
    and then starts the training by creating and running a `Trainer` instance.
    """
    os.environ["NCCL_TIMEOUT"] = "1200000"
    logger = logging.getLogger("training")

    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    print(vars(args))
    dataset_name = args.train_dataset
    seed = args.seed
    N = args.N
    B = args.B

    total_steps = args.total_steps
    encoder_spec = args.encoder_spec
    key_embd_src = args.key_embd_src
    use_data_aug = args.use_data_aug
    use_lr_decay = args.use_lr_decay
    use_cached_embd = args.use_cached_embd
    dataset_dir = args.dataset_dir
    model_dir_to_resume = args.model_dir_to_resume
    model_save_dir = args.model_save_dir
    sep_query_head = args.sep_query_head
    kb_size = args.kb_size
    dynamic_kb_size = args.dynamic_kb_size
    max_seq_len = args.max_seq_len

    if kb_size is not None and dynamic_kb_size is not None:
        raise ValueError("Can't specify kb_size and dynamic_kb_size. Use only one")

    kb_size = kb_size if kb_size is not None else dynamic_kb_size

    gradient_accm_step = args.gradient_accm_step

    length_invariance = args.length_invariance
    outlier_num = args.outlier_num
    multi_entities = args.multi_entities
    use_extended_qa = args.use_extended_qa
    kb_token_layer_frequency = args.kb_token_layer_frequency
    llm_type = args.llm_type
    hf_model_spec = args.hf_model_spec
    hf_token = args.hf_token
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")

    torch.manual_seed(seed)
    np.random.seed(seed)

    pathlib.Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    if Accelerator().is_main_process:
        wandb.init(
            # set the wandb project where this run will be logged
            project="kb-llm",
            # track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                'sep_query_head': sep_query_head,
                'kb_size': kb_size,
                'length_invariance': length_invariance,
                'dataset': dataset_name,
                'outlier_num': outlier_num,
                'multi_entities': multi_entities,
                'use_extended_qa': use_extended_qa,
                'kb_token_layer_frequency': kb_token_layer_frequency,
                'gradient_accm_step': gradient_accm_step,
                "encoder_spec": encoder_spec,
                "max_seq_len": max_seq_len,
            },
        )

    # Try to free up memory
    torch.cuda.empty_cache()

    if args.log_to_file:
        formatter = logging.Formatter(LOGFORMAT)
        f_handler = logging.FileHandler(os.path.join(model_save_dir, "log.txt"))
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    logger.info(f"Running on {device}")

    logger.info("Started training")
    logger.info(f"Saving to  {model_save_dir}")
    if sep_query_head:
        os.environ["SEP_QUERY_HEAD"] = "TRUE"
        logger.info("Having seperate query head for KB!")

    if length_invariance:
        os.environ["LENGTH_INVARIANCE"] = "TRUE"
        logger.info("Having seperate query head for KB!")

    os.environ["SCALE_FACTOR"] = ""
    
    key_embds, value_embds = None, None
    if use_cached_embd:
        # We load the pre-computed version stored on the disk rather
        # than computing them on the fly to make things faster
        logger.info(f"Using pre-computed {encoder_spec} embedding")
        key_embds, value_embds = _load_cached_embeddings(encoder_spec, dataset_dir, dataset_name, key_embd_src)

    prefix_string = get_prefix_str(args)
    logger.info(f"Experiment prefix {get_prefix_str(args)}")

    if use_extended_qa:
        dataset = json.load(open(os.path.join(dataset_dir, f"{dataset_name}_augmented.json")))
    else:
        dataset = json.load(open(os.path.join(dataset_dir, f"{dataset_name}.json")))

    training_set = dataset[:N]

    # Set up the LLM
    llm_model_spec = model_dir_to_resume if model_dir_to_resume else hf_model_spec

    resumed_step = 0 if not model_dir_to_resume else int(model_dir_to_resume.split("_")[-1])

    if llm_model_spec is None:
        raise ValueError("Either supply model_dir_to_resume or hf_model_spec")

    if hf_token is None and args.llm_type == "llama3":
        raise ValueError("Please supply HuggingFace token(hf_token) when loading model Llama weights from HuggingFace")

    # Tokenizer comes from the base model
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_spec,
        trust_remote_code=True,
        token=hf_token if hf_token and args.llm_type == "llama3" else None,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if args.llm_type == "llama3":
        model = KblamLlamaForCausalLM.from_pretrained(
            llm_model_spec,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            #token=hf_token,
        )
    elif args.llm_type == "phi3":
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            llm_model_spec,
            device_map=device,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    elif args.llm_type == "bitnet":
        model = KBLaMBitNetForCausalLM.from_pretrained(
            llm_model_spec,
            device_map=device,
            torch_dtype=torch.bfloat16, # BitNet uses bfloat16
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"LLM type {args.llm_type} not recognised")

    logger.info(model.config)  # type: ignore

    model.eval()  # type: ignore
    # freeze model
    for _, param in model.named_parameters():  # type: ignore
        param.requires_grad = False

    # Set up the encoder
    encoder = KBEncoder(
        encoder_name=encoder_spec,
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size  # type: ignore
        * (model.config.num_hidden_layers // kb_token_layer_frequency + 1),  # type: ignore
        frozen_base_model=True,
        device=device,
    )

    if model_dir_to_resume:
        encoder_dir = model_dir_to_resume + "_encoder"
        encoder.load_state_dict(torch.load(os.path.join(encoder_dir, "encoder.pt")))
        if args.llm_type == "bitnet":
            config_path = os.path.join(model_dir_to_resume, "kb_config_explicit.json")
        else:
            config_path = os.path.join(model_dir_to_resume, "kb_config.json")
        kb_config = KBLaMConfig.from_pretrained(config_path)
    else:
        kb_config = KBLaMConfig(
            sep_query_head=sep_query_head,
            kb_layer_frequency=kb_token_layer_frequency,
            kb_length_scaling=length_invariance,
            kb_max_train_triples=N,
        )

    encoder.train()

    kbretriever = KBRetriever(
        encoder,
        training_set,
        key_embds=key_embds,  # type: ignore
        value_embds=value_embds,  # type: ignore
    )

    logger.info("Model ready")

    # Get the training started
    llm_ckpt_name = f"{prefix_string}KeyFrom{key_embd_src}_{encoder_spec}_{dataset_name}_{llm_type}"

    trainer = Trainer(
        model,  # type: ignore
        kbretriever,
        tokenizer,
        kb_token_layer_frequency,
        total_steps,
        args.lr,
        device,
        use_lr_decay,
        kb_size,  # type: ignore
        llm_ckpt_name,
        model_save_dir,
        llm_type=llm_type,
        sep_query_head=sep_query_head,
        max_seq_len=max_seq_len,
    )

    logger.info(f"Number of trainable parameters: {_get_parameter_count(encoder):,}")

    trainer.train(
        training_set,
        B,
        gradient_accm_step,
        outlier_num,
        use_data_aug=use_data_aug,
        multi_entities=multi_entities,
        use_extended_qa=use_extended_qa,
        save_period=args.save_period,
        resumed_step=resumed_step,
        kb_config=kb_config,
    )


if __name__ == "__main__":
    main()
