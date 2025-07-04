import argparse

def parse_args():
    """Parses command-line arguments for the training script.

    This function defines and parses the command-line arguments required for
    training the model, including dataset specifications, hyperparameters,
    model configurations, and paths for saving and resuming training.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train_dataset",type=str,default="synthetic")
    parser.add_argument("--N", type=int, default=120000, help="Size of training set, select the first N samples for training")
    parser.add_argument("--B", type=int, default=10, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--sep_query_head", action=argparse.BooleanOptionalAction, help="Train a separate query head")
    parser.add_argument("--use_oai_embd", action="store_true", help="Use OpenAI embedding")
    parser.add_argument("--use_cached_embd", action="store_true", help="Choose to use pre-computed KV embeddings")
    parser.add_argument("--total_steps", type=int, default=20000, help="Total steps")
    parser.add_argument("--encoder_spec", type=str, default="OAI")
    parser.add_argument("--key_embd_src", type=str, default="key", choices=["key", "answer", "questions", None], help="Source of key embedding")
    parser.add_argument("--use_data_aug", action="store_true", help="Randomly pick templates for the question")
    parser.add_argument("--use_lr_decay", action="store_true")
    parser.add_argument("--dataset_dir", type=str, default="synthetic_data")
    parser.add_argument("--model_dir_to_resume", type=str, default=None, help="Checkpoint directory to resume training")
    parser.add_argument("--hf_model_spec", type=str, default="meta-llama/Llama-3.2-1B-Instruct", choices=["meta-llama/Meta-Llama-3-8B-Instruct", "microsoft/Phi-3-mini-4k-instruct", "meta-llama/Llama-3.2-1B-Instruct", "microsoft/bitnet-b1.58-2B-4T-bf16"])
    parser.add_argument("--hf_token", type=str,default=None,help="Huggingface token")
    parser.add_argument("--model_save_dir", type=str, default="output", help="Place to save the checkpoints")
    parser.add_argument("--kb_size", type=int, default=None, help="The size of the KB set size")
    parser.add_argument("--dynamic_kb_size", nargs=2, type=int, default=None, help="The size of the KB set size. Set a dynamic range for the kbsize specify min and max")
    parser.add_argument("--duplicate_true_kb", action=argparse.BooleanOptionalAction, default=True, help="Duplicate true entity's KB token")
    parser.add_argument("--length_invariance", action=argparse.BooleanOptionalAction, default=False, help="Scale the raw attention score")
    parser.add_argument("--outlier_num", type=int, default=1, help="Introduce questions without correct KB entites")
    parser.add_argument("--multi_entities", type=int, default=None, help="Introduce questions involving multiple entities")
    parser.add_argument("--use_extended_qa", action="store_true", help="Introduce QA with extended open-ended parts")
    parser.add_argument("--kb_token_layer_frequency", type=int, default=3, help="Introduce QA with extended open-ended parts")
    parser.add_argument("--gradient_accm_step", type=int, default=20, help="Introduce QA with extended open-ended parts")
    parser.add_argument("--verbose", action="store_true", help="Set logging to debug")
    parser.add_argument("--log_to_file", action="store_true", help="Log to file as well as stdout")
    parser.add_argument("--llm_type",type=str,default="llama3",choices=["llama3", "phi3", "bitnet"])
    parser.add_argument("--max_seq_len", type=int, default=None, help="Maximum sequence length")
    parser.add_argument("--save_period", type=int, default=100, help="Steps between checkpoints")
    return parser.parse_args()
