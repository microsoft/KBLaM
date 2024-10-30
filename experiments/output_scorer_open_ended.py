import argparse
import json
import re
from dataclasses import dataclass

import numpy as np

from kblam.gpt_session import GPT


@dataclass
class EvalExample:
    evidence: str
    question: str
    response: str
    score: float
    reason: str


def save_example(example: EvalExample, output_file: str) -> None:
    try:
        with open(output_file, "a+") as f:
            json.dump(example.__dict__, f)
            f.write("\n")
    except Exception as e:
        print(f"Error saving example.")
        print(e)


class Evaluator(GPT):
    def __init__(self, model, endpoint_url, seed, **kwargs) -> None:
        self.system_msg = """You are an AI system that evaluates the quality of generated response. Your goals is to return a score between 0 and 5
                                    indicating how accurate and useful the response is. An accrate and useful response should get a high score of 5."""
        self.prompt_open_ended = """
        A model is given a question about some information and evidence. 
        The question is composed of two parts, a part that involves repeating information in the evidence and a part that potentially involves open-ended thinking.
        Then the model generates a response.
        Evaluate the response based on how grounded it is given the evidence and how reasonable it is.
        Return an integer score and step by step explanation of how you arrived at the score.
        Score of 5 means the response is accurate, relevant and reasonable (in that it meets common sense).
        If the responce addresses the question and uses the evidence in a relevant way, it should get a high score of 5.
        Score of 0 means the response is inaccurate and irrelevant or model is hallucinating.
        Score between 0 and 5 means the response is partially correct and relevant.

        Example 1:
        Evidence: "The purpose of Alexandria is to extract knowledge."
        Question: "Describe the purpose of Alexandria and how it can benefit users."
        Model output: "The purpose of Alexandria is to extract knowledge, it can benefit users by providing a structured way to organize knowledge."
        Score: 5
        Reason: The model's response is accurate and relevant to the question and evidence, the open-ended part is reasonable.

        Example 2:
        Evidence: "The purpose of Alexandria is to extract knowledge."
        Question: "Describe the purpose of Alexandria and what can it extract."
        Model output: "The purpose of Alexandria is to extract knowledge, it can extract knowledge knowledge."
        Score: 5
        Reason: The model's response is accurate and relevant to the question and evidence.

        Example 3:
        Evidence: "GreatTool is an app that helps users to be more productive."
        Question: "Describe GreatTool and how it may affect the community."
        Model output: "GreatTool is an app that helps users to be more productive. It may affect the community by helping users to sleep better."
        Score: 3
        Reason: The model's response is accurate and relevant to the question and evidence but it is not very reasonable.


        Example 4:
        Evidence: "GreatTool is an app that helps users to be more productive."
        Question: "Describe GreatTool and how it may affect the community."
        Model output: "GreatTool is an app that helps users to be more productive. It may affect the community by helping users to organize their tasks and manage their time better improving their productivity."
        Score: 5
        Reason: The model's response is accurate and relevant to the question and evidence and the open ended part is sensible and reasonable.

        Example 5:
        Evidence: "GreatTool is an app that helps users to be more productive."
        Question: "Tell me the description of GreatTool and what can it help users to achieve."
        Model output: "GreatTool is an app that helps users to be more productive. It can help users to organize their tasks and manage their time better improving their productivity."
        Score: 5
        Reason: The model's response is accurate and relevant to the question and evidence.

        Example 6:
        Evidence: "GreatTool is an app that helps users to be more productive."
        Question: "Describe GreatTool and how it may affect the community."
        Model output: "GreatTool is great tool with many feature"
        Score: 0
        Reason: The model's response is not accurate and doesn't answer the question.

        Example 7:
        Evidence: "GreatTool is an app that helps users to be more productive."
        Question: "Describe GreatTool and how it may affect the community."
        Model output: "GreatTool is an app that helps users to be more productive, it improves community income level."
        Score: 3
        Reason: The model's response is accurate but is not very reasonable.
        """
        self.prompt_open_ended += (
            "\n Score the following responce: \n evidence: {0}, question: {1} and \n model response: {2}"
        )

        self.seed = seed
        super().__init__(model, endpoint_url, **kwargs)

    def evaluate_open_ended(self, prompt, evidence: str, question: str, response: str) -> str:
        prompt = prompt.format(evidence, question, response)
        return self.generate_response(prompt)

    def evaluate_output_batch(self, examples: list[str]) -> list[str]:
        score_pattern = r"Score: (.+)"
        reason_pattern = r"Reason: (.+)"

        eval_examples = []
        for example in examples:
            try:
                evidence_start = example.find("Evidence:")
                question_start = example.find("Question:")
                model_output_start = example.find("Model output:")

                # Extract the parts based on the indices
                evidence = example[evidence_start:question_start].strip()
                question = example[question_start:model_output_start].strip()
                model_output = example[model_output_start:].strip()

                eval_example = self.evaluate_open_ended(self.prompt_open_ended, evidence, question, model_output)
                score = float(re.search(score_pattern, eval_example).group(1).strip())
                reason = re.search(reason_pattern, eval_example).group(1).strip()
                eval_example = EvalExample(evidence, question, model_output, score, reason)
                eval_examples.append(eval_example)

                save_example(eval_example, args.output_file)

            except Exception as e:
                print(f"Error evaluating example.")
                print(e)
                return eval_examples


def parser_args():
    parser = argparse.ArgumentParser(description="GPT Session")
    parser.add_argument("--model", type=str, default="GPT4", help="The model to use.")
    parser.add_argument("--endpoint_url", type=str, help="The endpoint url.")
    parser.add_argument(
        "--predictions_file",
        type=str,
        default="ScaleFactor_100FancyInstruction_lr_0.0001_kblayer_frequency_1_encoder_OAI_ckptidx_20000testset_gpt_data_kbsize_100_evalmode_icl_seed_1.txt",
        help="The file containing the model predictions.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="eval_examples_open_ended_icl.json",
        help="The output file to save the examples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parser_args()
    with open(args.predictions_file, "r") as f:
        examples = f.read()
        examples = examples.split("-------")

    eval = Evaluator(args.model, args.endpoint_url, args.seed)
    eval_examples = eval.evaluate_output_batch(examples)
    mean_score = np.mean([example.score for example in eval_examples])
    print(f"Mean score: {mean_score}")
