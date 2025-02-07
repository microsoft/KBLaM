# Dataset Card for Enron extracted knowledgebase

The Enron extracted knowledgebase consists of triples that were extracted using a small language model from the Enron database.

## Dataset Details

### Dataset Description

The triples in this knowledge base were extracted from the Enron email dataset using a small language model trained for entity extraction, then were clustered for de-duplication of entities and converted into triples where the relations were description, objective, and purpose.

- **Curated by:** Microsoft Research
- **Language(s) (NLP):** English
- **License:** MIT

## Uses

The dataset is intended to be used for the training and evaluation of grounded LLMs. The dataset can also be used for needle-in-the-haystack retrieval tasks, by augmenting the questions with a number of noise triples.

### Direct Use

Research model training and evaluation.

### Out-of-Scope Use

The dataset may not be useful as a real-world knowledge base as the triples were extracted using an automated system.

## Dataset Structure

A list of JSONs, each with the following properties:

`name`: name of entity
`description_type`: the name of the property
`description`: the value of the property
`Q`: A question based on the triple
`A`: An answer based on the triple
`key_string`: The key used in KBLaM (created with a template of "The {property name} of {entity name}")

## Dataset Creation

### Curation Rationale

The data was created to allow for the evaluation of knowledge-base augmented LLMs on real-world data.

### Source Data

Enron email dataset

#### Data Collection and Processing

The entities were extracted using a generative SLM fine-tuned on the task, and linked using the project Alexandria entity linker for disambiguation.

#### Who are the source data producers?

The Enron email dataset is provided by CMU, who sourced it from Enron.

#### Personal and Sensitive Information

No additional personal information is contained over the Enron email dataset.

## Bias, Risks, and Limitations

This dataset reflects the biases from the Enron email dataset, and may also be limited by the capabilities of the extraction process. This dataset only contains one description, objective, and purpose for each entity, when many more were extracted. This means that its use as a complete knowledgebase is limited.

### Recommendations

Due to the limitations as a complete knowledge base, it is recommended that this dataset is used for the evaluation of knowledgebase-augmented models only.

## Dataset Card Contact

t-isazawat@microsoft.com
