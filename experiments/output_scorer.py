import argparse
import json
from dataclasses import dataclass

import numpy as np

from kblam.gpt_session import GPT


@dataclass
class EvalExample:
    text: str
    true_answer: str
    score: float


def save_example(example: EvalExample, output_file: str) -> None:
    try:
        with open(output_file, "a+") as f:
            json.dump(example.__dict__, f)
            f.write("\n")
    except Exception as e:
        print(f"Error saving example.")
        print(e)


class Evaluator(GPT):
    def __init__(self, model, endpoint_url, **kwargs) -> None:
        self.system_msg = """You are an AI system that evaluates the quality of generated text. 
                                You will be given a text and a ground truth answer, your goals is to return a score between 0 and 1."""
        self.prompt = """ Given a text and a ground truth answer, evaluate the quality of the text.
                            Return a score of 1 if the text is exactly the same as the ground truth answer,
                            Return a score of 0 if the text is completely wrong,
                            Return a score between 0 and 1 if the text is partially correct. A more correct text should have a higher score.
                            Do NOT generate anything else.
                            Example:

                            Model output: "The sky is blue."
                            True answer: "The sky is blue."
                            Score: 1

                            Example 2:
                            Model output: "The color of Alexandria is blue."
                            True answer: "The color of Alexandria is green."
                            Score: 0

                            Example 3:
                            Model output: "The purpose of Alexandria is to extract knowledge."
                            True answer: "The color of Alexandria is to discover and organize knowledge into a structured form."
                            Score: 0.9

                            **Important**: Only generate a number.
                            """
        self.prompt += "\n Score the following text: \n model prediction: {0}, \n true answer: {1}"
        self.seed = 42
        super().__init__(model, endpoint_url, **kwargs)

    def evaluate_output(self, prompt: str, text: str, true_answer: str) -> str:
        prompt = self.prompt.format(text, true_answer)
        score = self.generate_response(prompt)
        example = EvalExample(text, true_answer, float(score))
        return example

    def evaluate_output_batch(self, examples: list[str]) -> list[str]:
        eval_examples = []
        for example in examples:
            try:
                text = example.split("True answer:")[0].replace("Model output:", "").strip()
                true_answer = example.split("True answer:")[1].strip()
                eval_example = self.evaluate_output(self.prompt, text, true_answer)
                eval_examples.append(eval_example)
            except Exception as e:
                print(f"Error evaluating example.")
                print(e)
        return eval_examples


def parser_args():
    parser = argparse.ArgumentParser(description="GPT Session")
    parser.add_argument("--model", type=str, default="GPT4", help="The model to use.")
    parser.add_argument("--endpoint_url", type=str, help="The endpoint url.")
    parser.add_argument("--predictions_file", type=str, default="llama.txt", help="The input file with examples.")
    parser.add_argument(
        "--output_file", type=str, default="eval_examples1.json", help="The output file to save the examples."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parser_args()
    with open(args.predictions_file, "r") as f:
        examples = f.read()
        examples = examples.split("-------")

    eval = Evaluator(args.model, args.endpoint_url)
    eval_examples = eval.evaluate_output_batch(examples)
    for example in eval_examples:
        save_example(example, args.output_file)

    mean_score = np.mean([example.score for example in eval_examples])
    print(f"Mean score: {mean_score}")
