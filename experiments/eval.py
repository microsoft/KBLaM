"""Script for evaluating KB models"""

import nltk
from transformers import logging

from eval.args import parse_args
from eval.generation import eval_generate
from eval.accuracy import eval_accuracy_cli, run_accuracy_evalution
from eval.refusal import eval_refusal
from eval.standard import eval_standard


nltk.download("wordnet")
logging.set_verbosity_warning()


def main():
    """The main entry point for the evaluation script.

    This function parses command-line arguments to determine which evaluation
    mode to run (generation, accuracy, refusal, or standard) and then calls
    the corresponding evaluation function.
    """
    args = parse_args()
    print(args)
    if args.command == "generation":
        eval_generate(args)
    elif args.command == "accuracy":
        eval_accuracy_cli(args)
    elif args.command == "acc_results":
        run_accuracy_evalution(args)
    elif args.command == "refusal":
        eval_refusal(args)
    elif args.command == "standard":
        eval_standard(args)
    else:
        raise ValueError(f"command {args.command} not recognised")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()