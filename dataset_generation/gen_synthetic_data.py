import argparse
import json
import os
import re
from itertools import product

from tqdm import tqdm

from kblam.gpt_session import GPT
from kblam.utils.data_utils import DataPoint, Entity, save_entity


def construct_prompts(entity: DataPoint) -> tuple[str, str, str]:
    """Take in an entity, return constructed question, anwer and key name."""
    Q = "What is the {} of {}?".format(entity.description_type, entity.name)
    A = f"The {entity.description_type} of {entity.name} is {entity.description}."
    key_string = f"the {entity.description_type} of {entity.name}"
    return Q, A, key_string


class SyntheticDataGenerator(GPT):
    def __init__(self, model, endpoint_url, **kwargs) -> None:
        self.system_prompt = """You are a AI system that generates synthetic data examples in JSON format."""

        self.entity_format_prompt = """
            \nMake sure to generate a single data point in the following JSON format:
            {
                "name": "{name}",
                "description": "{description}",
                "objectives": "{objectives}",
                "purpose": "{purpose}"
            }
        """

        self.prompt_2nd_phase = (
            """
            Now for each of the names generated, generate a short desciption, short objectives, and a purpose for the data.
            Please ensure that the generated contents has **LOW** correlation with the name.
        """
            + self.entity_format_prompt
            + " Do **NOT** generate anything else."
        )

        self.prompt_3rd_phase = (
            """
            Now for each of the name, description, objective and purpose generated, make their text style more diverse using a mixture of formal and informal language.
        """
            + self.entity_format_prompt
        )

        self.idea_sources = [
            "greek letters",
            "fiction characters",
            "famous rock bands",
            "birds",
            "animals",
            "natural phenomena",
            "physical locations",
            "artist names",
            "classical music",
            "musical instruments",
            "music genres",
            "art styles",
            "ancient Roman concepts",
            "Hindu myths",
            "Cthulhu Mythos",
            "real-world company names",
            "mythological creatures",
            "planets and stars",
            "historical figures",
            "literary genres",
            "botanical names",
            "famous landmarks",
            "scientific concepts",
            "space missions",
            "inventions",
            "philosophical terms",
            "chemical elements",
            "famous scientists",
            "marine life",
            "mythological places",
        ]

        self.name_types = [
            "education company",
            "tech company",
            "car company",
            "entertainment company",
            "construction company",
            "retail company",
            "finance company",
            "healthcare company",
            "restaurant",
            "hotel",
            "github repo",
            "project",
            "meeting room",
            "building",
            "lab",
            "airline",
            "textbook",
            "website",
            "personal blog",
            "gaming company",
            "consulting firm",
            "biotech company",
            "app",
            "software tool",
            "bookstore",
            "e-commerce site",
            "social media platform",
            "fitness brand",
            "fashion brand",
            "non-profit organization",
        ]

        super().__init__(model, endpoint_url, **kwargs)

    def get_instructions(self):
        return [
            f"Please randomly generate a {name_type} name innovated by {idea_type}."
            "The generated name should be of diverse style and length. A valid name should consist of a single word (such as Alexandria) or multiple words (such as Microsoft Office or Theta-Phoenix Entertainment). "
            for (name_type, idea_type) in product(self.idea_sources, self.name_types)
        ]

    def generate_entity(self, instruction: str) -> Entity:
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": instruction},
        ]
        gpt_output = self.api_call_chat(prompt)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": gpt_output},
            {"role": "user", "content": self.prompt_2nd_phase},
        ]
        gpt_output = self.api_call_chat(messages)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": gpt_output},
            {"role": "user", "content": self.prompt_2nd_phase},
            {"role": "assistant", "content": gpt_output},
            {"role": "user", "content": self.prompt_3rd_phase},
        ]

        gpt_output = self.api_call_chat(messages)
        entity = Entity(**json.loads(gpt_output))
        return entity

    def post_process_data(self, entity_list: list[Entity]) -> list[DataPoint]:
        dataset = []
        keywords = {"description", "objectives", "purpose"}

        for entity in entity_list:
            for keyword in keywords:
                datapoint = DataPoint(
                    name=entity.name,
                    description_type=keyword.lower(),
                    description=getattr(entity, keyword),
                )
                datapoint.Q, datapoint.A, datapoint.key_string = construct_prompts(datapoint)
                dataset.append(datapoint)

        return dataset

    def augmenta_data_with_synthetic_QA(self, dataset: list[DataPoint]) -> list[DataPoint]:
        self.system_prompt = """You are given a question and answer pair, please extend the question to be open-ended and generate a short answer. 
                                For example, you could generate "What is the objective of xxx and what do you think of it?"
                                Make sure the answer is **only** based on information provided from the QA pair. In addition, please generate in the format of:
                                Q: ...
                                A: ... 
                            """

        for data in dataset:
            try:
                prompt = "Generate an extended Q and an A for this pair: " + f"Q: {data.Q}\nA: {data.A}"
                gpt_output = self.generate_response(prompt)
                extended_q = re.findall(r"Q: (.*)", gpt_output)[0]
                extended_a = re.findall(r"A: (.*)", gpt_output)[0]
                data.extended_Q = extended_q
                data.extended_A = extended_a
            except Exception as e:
                print(f"Error augmenting Q&A.")
                print(e)
                continue
        return dataset


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--endpoint_url", type=str)
    parser.add_argument("--output_path", type=str, default="dataset")
    parser.add_argument("--raw_output_file", type=str, default="synthetic_data_raw.json")
    parser.add_argument("--output_file", type=str, default="synthetic_data_QA.json")
    parser.add_argument("--augmented_output_file", type=str, default="synthetic_data_QA_augmented.json")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()

    data_generator = SyntheticDataGenerator(args.model_name, args.endpoint_url)

    os.makedirs(args.output_path, exist_ok=True)

    entity_list = []
    for seed in range(1):
        data_generator.set_seed(seed)
        for instruction in tqdm(data_generator.get_instructions()):
            try:
                response = data_generator.generate_entity(instruction)
            except Exception as e:
                print(f"Error generating entity.")
                print(e)
                continue
            save_entity(response, args.raw_output_file)
            entity_list.append(response)

    dataset = data_generator.post_process_data(entity_list)
    for data in dataset:
        save_entity(data, args.output_file)

    dataset = data_generator.augmenta_data_with_synthetic_QA(dataset)
    for data in dataset:
        save_entity(data, args.augmented_output_file)
