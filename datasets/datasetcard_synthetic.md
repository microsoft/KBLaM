# Dataset Card for KBLaM synthetic grounded questions dataset

The KBLaM synthetic grounded questions dataset consists of a synthetic, GPT-4 generated knowledgebase of triples and a number of factual questions on this dataset.

## Dataset Details

### Dataset Description

- **Curated by:** Microsoft Research
- **Language(s) (NLP):** English
- **License:** MIT

## Uses

The dataset is intended to be used for the training and evaluation of grounded LLMs. The dataset can also be used for needle-in-the-haystack retrieval tasks, by augmenting the questions with a number of noise triples.

### Direct Use

Research model training and evaluation.

### Out-of-Scope Use

The dataset will not work well as a real-world knowledgebase as the triples are entirely synthetic.

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

The data was created using GPT to allow for training and evaluation of knowledge-base augmented LLMs.

### Source Data

N/A - the data is entirely synthetic, produced by GPT-4

#### Data Collection and Processing

The data was created synthetically using GPT-4.

#### Who are the source data producers?

The data was created synthetically using GPT-4.

#### Personal and Sensitive Information

The data was created synthetically using GPT-4, so personal data is unlikely.

## Bias, Risks, and Limitations

As the data was created by GPT-4, the dataset's distribution will be biased towards GPT-4's builtin biases, and this should be taken as a limitation when creating or evaluating a model.

### Recommendations

Any models should be evaluated also on other, human-created, datasets to ensure good real-world performance.

## Dataset Card Contact

t-isazawat@microsoft.com
