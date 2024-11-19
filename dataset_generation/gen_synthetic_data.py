import argparse
import json
import os
import re
from itertools import product

from tqdm import tqdm
from transformers import AutoModelForCausalLM

from kblam.gpt_session import GPT
from kblam.utils.data_utils import DataPoint, Entity, save_entity


def construct_prompts(entity: DataPoint) -> tuple[str, str, str]:
    """Take in an entity, return constructed question, anwer and key name."""
    Q = "What is the {} of {}?".format(entity.description_type, entity.name)
    A = f"The {entity.description_type} of {entity.name} is {entity.description}."
    key_string = f"the {entity.description_type} of {entity.name}"
    return Q, A, key_string


class SyntheticDataGenerator(GPT):
    def __init__(self, model: AutoModelForCausalLM, endpoint_url: str, **kwargs) -> None:
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
            Now for each of the names generated, generate a short desciption, short objectives, and a purpose for the data point.
            Please ensure that the generated contents has **LOW** correlation with the name.
            Make the data point styles diverse using a mixture of formal and informal language.
        """
            + self.entity_format_prompt
            + " Do **NOT** generate anything else."
        )

        self.idea_sources = [
            "software companies",
            "tech companies",
            "software tools",
            "greek letters",
            "product reviews",
            "product releases",
            "work-related concepts",
            "work-related documents",
            "document types",
            "financial terms",
            "legal terms",
            "medical terms",
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
            "political figures",
            "literary genres",
            "botanical names",
            "famous landmarks",
            "scientific concepts",
            "space missions",
            "inventions",
            "philosophical terms",
            "chemical elements",
            "famous scientists",
            "famous mathematicians",
            "famous authors",
            "marine life",
            "mythological places",
            "famous battles",
            "sports teams",
            "sport events",
            "food and drinks",
        ]

        self.data_types = [
            "person name",
            "idea",
            "team",
            "meeting",
            "event",
            "location",
            "document",
            "presentation",
            "meeting",
            "conference",
            "workshop",
            "database",
            "organization",
            "tech company",
            "car company",
            "entertainment company",
            "construction company",
            "retail company",
            "finance company",
            "healthcare company",
            "restaurant",
            "hotel",
            "museum",
            "university",
            "educational institution",
            "government agency",
            "hospital",
            "github repo",
            "project",
            "meeting room",
            "building",
            "product",
            "lab",
            "airline",
            "textbook",
            "tv show",
            "music album",
            "website",
            "personal blog",
            "gaming company",
            "game" "movie studio",
            "consulting firm",
            "biotech company",
            "app",
            "software tool",
            "bookstore",
            "coffee shop",
            "bar",
            "e-commerce site",
            "social media platform",
            "fitness brand",
            "fashion brand",
            "beauty brand",
            "food brand",
            "drink brand",
            "sports brand",
            "travel brand",
            "non-profit organization",
            "political party",
        ]

        super().__init__(model, endpoint_url, **kwargs)

    def get_instructions(self):
        return [
            f"Please randomly generate a {name_type} name innovated by or associated with {idea_type}."
            "The generated name should be of diverse style and length. A valid name should consist of a single word (such as Alexandria or Microsoft) or multiple words (such as Microsoft Office or Theta-Phoenix Entertainment). "
            for (name_type, idea_type) in product(self.idea_sources, self.data_types)
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

        gpt_output = self.api_call_chat(messages)
        entity = Entity(**json.loads(gpt_output))
        return entity

    def generate_related_data(self, entity: Entity) -> Entity:
        instruction = f"Generate a person name related to the entity {entity.name} with description {entity.description}."
        instruction += "The person needs to be associated with the entity in some way. e.g. they work in the company or they are a character in the book."
        instruction += f"Make sure the entity is in the format of {self.entity_format_prompt}"

        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": instruction},
        ]

        gpt_output = self.api_call_chat(prompt)
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
                print(gpt_output)
                extended_q = re.findall(r"Q: (.*)", gpt_output)[0]
                extended_a = re.findall(r"A: (.*)", gpt_output)[0]
                data.extended_Q = extended_q
                data.extended_A = extended_a
            except Exception as e:
                print(f"Error augmenting Q&A.")
                print(e)
                continue
        return dataset
    

    def perturb_names(self, dataset: list[DataPoint]):
        for data in dataset:
            try:
                prompt = f"Perturb the names in the queries of the dataset (e.g. Margaret Thatcher -> Maggie Thatcher or Microsoft Research to MSR) for data point with name {data.name}."
                prompt += f"Return the question {data.Q} with the perturbed name. Make sure the perturbation is valid. Do NOT generate anything else."
                gpt_output = self.generate_response(prompt)
                data.Q = gpt_output

            except Exception as e:
                print(f"Error perturbing the names in the queries.")
                print(e)
                continue
        return dataset




def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--endpoint_url", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="dataset")
    parser.add_argument("--generate_related_people", type=bool, default=True)
    parser.add_argument("--raw_output_file", type=str, default="synthetic_data_raw.json")
    parser.add_argument("--output_file", type=str, default="synthetic_data_QA.json")
    parser.add_argument("--perturbed_output_file", type=str, default="perturbed_output_file")
    parser.add_argument("--augmented_output_file", type=str, default="synthetic_data_QA_augmented.json")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()

    data_generator = SyntheticDataGenerator(args.model_name, args.endpoint_url)

    os.makedirs(args.output_path, exist_ok=True)

    raw_output_file = os.path.join(args.output_path, args.raw_output_file)

    if os.path.exists(raw_output_file):
        # skip entities creation if it's already generated
        with open(raw_output_file, "r") as file:
            entity_list = [Entity(**json.loads(line)) for line in file]

    else:
        entity_list = []
        for seed in range(1):
            data_generator.set_seed(seed)
            for instruction in tqdm(data_generator.get_instructions()):
                try:
                    entity = data_generator.generate_entity(instruction)
                except Exception as e:
                    print(f"Error generating entity.")
                    print(e)
                    continue
                save_entity(entity, raw_output_file)
                entity_list.append(entity)

                if args.generate_related_people:
                    try:
                        response = data_generator.generate_related_data(entity)
                    except Exception as e:
                        print(f"Error generating entity.")
                        print(e)
                        continue
                    save_entity(response, raw_output_file)
                    entity_list.append(response)

    QA_output_file = os.path.join(args.output_path, args.output_file)

    if os.path.exists(QA_output_file):
        with open(QA_output_file, "r") as file:
            dataset = [DataPoint(**json.loads(line)) for line in file]
    else:
        dataset = data_generator.post_process_data(entity_list)
        for data in dataset:
            save_entity(data, QA_output_file)

    perturbed_dataset = data_generator.perturbe_names(dataset)

    for data in perturbed_dataset:
        save_entity(data, os.path.join(args.output_path, args.perturbed_output_file))

    dataset = data_generator.augmenta_data_with_synthetic_QA(dataset)
    for data in dataset:
        save_entity(data, os.path.join(args.output_path, args.augmented_output_file))
