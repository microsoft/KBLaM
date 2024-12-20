import json
from dataclasses import dataclass

import numpy as np


@dataclass
class Entity:
    name: str
    description: str
    objectives: str
    purpose: str


@dataclass
class DataPoint:
    name: str
    description_type: str
    description: str
    Q: str = None
    A: str = None
    key_string: str = None
    extended_Q: str = None
    extended_A: str = None


def save_entity(pair: Entity | DataPoint, output_file: str) -> None:
    """Save a JSON entity to a file."""
    try:
        with open(output_file, "a+") as f:
            json.dump(pair.__dict__, f)
            f.write("\n")
    except Exception as e:
        print("Error saving entity.")
        print(e)


def load_entities(inout_file: str) -> list[Entity | DataPoint]:
    """Load entities from a file."""
    entities = []
    try:
        with open(inout_file, "r") as f:
            for line in f:
                entity = json.loads(line)
                entities.append(entity)
    except Exception as e:
        print("Error loading entities.")
        print(e)
    return entities


def get_i_dont_know_ans():
    return "I am sorry I cannot find relevant information in the KB."


def augment_row(row: dict[str, str]) -> list[dict[str, str]]:
    """Augment an entity with questions from pre-defined templates."""
    templates = [
        "What {} does {} have?",
        "What is the {} of {}?",
        "Tell me about the {} of {}.",
        "Can you let me know the {} of {}?",
        "Can you inform me about the {} of {}?",
        "Describe the {} of {}.",
        "What details can you share about the {} of {}?",
        "What kind of {} does {} have?",
        "Provide details on the {} of {}.",
        "What features does the {} of {} include?",
        "Can you elaborate on the {} of {}?",
        "How would you describe the {} of {}?",
        "What can you tell me about the {} characteristics of {}?",
        "Can you explain the {} of {}?",
        "What insights can you provide about the {} of {}?",
        "What should I know about the {} of {}?",
    ]
    dtype = row["description_type"]
    name = row["name"]
    tid = np.random.randint(0, len(templates))
    return templates[tid].format(dtype, name)


def generate_multi_entity_qa(
    names: list[str], properties: list[str], answers: list[str]
) -> tuple[str, str]:
    """Generate a question-answer pair for multiple entities."""
    templates = [
        "What is {}?",
        "Tell me {}.",
        "Can you let me know {}?",
        "Can you inform me {}?",
        "Describe {}.",
        "Explain {}.",
        "Could you describe the {}?",
        "What can you tell me about {}?",
        "Could you provide information on {}?",
        "Please enlighten me about {}.",
        "Can you clarify {} for me?",
        "Could you give me a detailed description of {}?",
        "I need more information on {}.",
    ]
    template_idx = np.random.randint(0, len(templates))
    question_body = ""
    for name, property in zip(names[:-1], properties[:-1]):
        question_body += f"the {property} of {name},"
    question_body += f" and the {properties[-1]} of {names[-1]}"
    answer_str = ""
    for answer, name, property in zip(answers, names, properties):
        answer_str += f"The {property} of {name} is {answer}; "
    return templates[template_idx].format(question_body), answer_str.strip()
