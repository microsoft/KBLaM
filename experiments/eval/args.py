import argparse

def parse_args():
    """Parses command-line arguments for the evaluation script.

    This function sets up and parses command-line arguments for various evaluation
    modes, including generation, accuracy, refusal, and standard evaluation. It uses
    subparsers to handle different sets of arguments for each command.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluation script")

    # Add arguments that will be shared across all subcommands
    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument(
        "--dataset_dir", type=str, help="Directory containing the dataset"
    )
    parent_parser.add_argument(
        "--encoder_dir", type=str, help="Directory containing the encoder model"
    )
    parent_parser.add_argument(
        "--encoder_spec",
        type=str,
        default="OAI",
        help="Specification for the encoder model",
    )
    parent_parser.add_argument(
        "--fancy_instruction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use fancy instructions",
    )
    parent_parser.add_argument(
        "--kb_layer_frequency",
        type=int,
        default=3,
        help="Frequency of knowledge base layers",
    )
    parent_parser.add_argument(
        "--kb_scale_factor",
        type=int,
        default=None,
        help="Scaling factor for knowledge base",
    )
    parent_parser.add_argument(
        "--kb_size", type=int, default=200, help="Size of the knowledge base"
    )
    parent_parser.add_argument(
        "--llm_base_dir",
        type=str,
        help="llm to load, can be HF location or local directory",
    )
    parent_parser.add_argument(
        "--llm_type",
        type=str,
        default="phi3",
        choices=["llama3", "phi3", "bitnet"],
        help="Type of language model to use",
    )
    parent_parser.add_argument(
        "--model_dir", type=str, help="Directory containing the model"
    )
    parent_parser.add_argument("--save_dir", type=str, help="Directory to save outputs", default="output")
    parent_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parent_parser.add_argument(
        "--test_dataset", type=str, help="Source of test KB (assumes KV pair format)"
    )
    parent_parser.add_argument(
        "--precomputed_embed_keys_path", type=str, help="Path to precomputed key embeddings"
    )
    parent_parser.add_argument(
        "--precomputed_embed_values_path",
        type=str,
        help="Path to precomputed value embeddings",
    )
    parent_parser.add_argument(
        "--query_head_path", type=str, default="", help="Path to load KB head from"
    )
    parent_parser.add_argument(
        "--topk_size",
        type=int,
        default=-1,
        help="The number of top-k entities to retrieve from the knowledge base.",
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Create the parser for the generation command
    gen_parser = subparsers.add_parser(
        "generation", help="Evaluate generation.", parents=[parent_parser]
    )
    gen_parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["kb", "icl", "zeroshot"],
        default="kb",
        help="Evaluation mode: knowledge base, in-context learning, or zero-shot",
    )
    gen_parser.add_argument(
        "--exp_config_name",
        type=str,
        default="generation_results",
        help="Name of the experiment configuration",
    )
    gen_parser.add_argument(
        "--kb_token_layer_frequency",
        type=int,
        default=None,
        help="Frequency of knowledge base token layers",
    )
    gen_parser.add_argument(
        "--multi_entites",
        type=int,
        default=-1,
        help="Number of entities to process (-1 for unlimited)",
    )
    gen_parser.add_argument(
        "--no_outlier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use checkpoints trained without outliers",
    )
    gen_parser.add_argument(
        "--remove_sorry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Filter out "sorry" answers from the output',
    )

    # Create the parser for the accuracy command
    acc_parser = subparsers.add_parser(
        "accuracy", parents=[parent_parser], help="Evaluate accuracy"
    )

    acc_parser.add_argument(
        "--attn_save_dir", type=str, default="", help="Directory to save attention masks"
    )
    acc_parser.add_argument(
        "--exp_config_name",
        type=str,
        default="accuracy_results",
        help="Name of the experiment configuration",
    )
    acc_parser.add_argument(
        "--fancy_question",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable fancy question format",
    )
    acc_parser.add_argument(
        "--log_save_dir", type=str, help="Directory to save accuracy results"
    )
    acc_parser.add_argument(
        "--test_batch_size", type=int, default=50, help="Batch size for testing"
    )
    acc_parser.add_argument(
        "--use_shift_match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable shift matching",
    )

    # Create the parser for the accuracy eval
    subparsers.add_parser(
        "acc_results", parents=[acc_parser], help="run accuracy eval", add_help=False
    )


    # Create the parser for the refusal command
    ref_parser = subparsers.add_parser(
        "refusal", parents=[parent_parser], help="Evaluate refusal"
    )
    ref_parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["kb", "icl", "zeroshot"],
        default="kb",
        help="Evaluation mode: knowledge base, in-context learning, or zero-shot",
    )
    ref_parser.add_argument(
        "--exp_config_name",
        type=str,
        default="refusal_results",
        help="Name of the experiment configuration",
    )
    ref_parser.add_argument(
        "--kb_token_layer_frequency",
        type=int,
        default=None,
        help="Frequency of knowledge base token layers",
    )
    ref_parser.add_argument(
        "--multi_entites",
        type=int,
        default=-1,
        help="Number of entities to process (-1 for unlimited)",
    )
    ref_parser.add_argument(
        "--no_outlier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use checkpoints trained without outliers",
    )
    ref_parser.add_argument(
        "--remove_sorry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Filter out "sorry" answers from the output',
    )

    # Create the parser for the standard command
    basic_parser = subparsers.add_parser(
        "standard", parents=[parent_parser], help="Evaluate basic performance"
    )
    basic_parser.add_argument(
        "--attn_summary_save_dir",
        type=str,
        default="",
        help="Directory to save attention masks",
    )
    basic_parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["kb", "icl", "zeroshot"],
        default="kb",
        help="Evaluation mode: knowledge base, in-context learning, or zero-shot",
    )
    basic_parser.add_argument(
        "--exp_config_name",
        type=str,
        default="basic_results",
        help="Name of the experiment configuration",
    )
    basic_parser.add_argument(
        "--exp_config_str", type=str, help="Experiment configuration string"
    )
    basic_parser.add_argument(
        "--kb_token_layer_frequency",
        type=int,
        default=None,
        help="Frequency of knowledge base token layers",
    )
    basic_parser.add_argument(
        "--no_outlier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use checkpoints trained without outliers",
    )
    basic_parser.add_argument(
        "--sample_size", default=5, type=int, help="Number of samples to process"
    )
    basic_parser.add_argument(
        "--subset_size", default=100, type=int, help="Size of the data subset to use"
    )
    return parser.parse_args()
