# KBLaM - Knowledge Base Augmented Language Models

This repo contains the official implementation of [KBLaM: Knowledge Base Augmented Language Models](https://arxiv.org/abs/2410.10450).

Authors: Xi Wang, Liana Mikaelyan, Taketomo Isazawa, Mathew Salvaris, James Hensman.

KBLaM is a new method for augmentating LLMs with external knowledge. 
Unlike Retrieval-Augmented Generation, KBLAM eliminates external
retrieval modules, and unlike in-context learning, its computational overhead scales linearly with KB size rather than quadratically.

## Supported Models
The following models from HuggingFace hub are currently supported:
 - [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
 - [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
 - [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

To add support for new model types, you will need to update the model processing scripts to incorporate an adapter similar to `llama_model.py` in `src/kblam/models`.

## Setting up

Install the kblam package with 

```
pip install -e .
```

To use Llama models, you will need to generate a token from HuggingFace and use it to log in:

```
pip install huggingface_hub
huggingface-cli login
```

The experiments in the paper can be replicated by running the scripts in `./experiments`.


## Dataset Construction

To run the synthetic dataset construction, you will need a valid Azure OpenAI endpoint. 

To construct a synthetic KB and question-answer pairs use `dataset_generation/gen_synthetic_data.py`

The question-answer pairs are constructed in the form:

```
What is the description of {entity_name}?
The description of {entity_name} is {description}.
```

To generate KB embeddings, use `dataset_generation/generate_kb_embeddings.py`.
The embeddings we current support are [text-embedding-ada-002](https://openai.com/index/new-and-improved-embedding-model/) and [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).


## Training

To train the model, run the following (with the appropriate arguments):

```
python train.py --dataset avocado_new --N 120000 --B 20 --steps 601  --encoder_spec OAI --use_oai_embd --key_embd_src key --use_data_aug
```

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## FAQ

### What is KBLaM?

KBLaM is a method to enhance a transformer-based LLM to augment it with knowledge. It consists of a base LLM, and some adapters that we train to transform the knowledge base to special knowledge tokens that the LLM ingests. In particular, because we only train adapters over the knowledge part, the base LLM is completely unmodified with regards to text input. If given no knowledge base, the model outputs the exact same thing as the base model for any given input.

### What can KBLaM do?

KBLaM can, in addition to the base LLM’s capabilities, also attend over the knowledge base to answer questions in a grounded manner.

### What is/are KBLaM’s intended use(s)?

The model is intended to be used for research.

### How was KBLaM evaluated? What metrics are used to measure performance?

KBLaM was evaluated on accuracy of retrieval from the knowledge base, its refusal rate (how often it correctly said that it didn’t have the requisite information to answer the question), and precision and recall on how well the answers aligned with the correct answers given the knowledge base.

### What are the limitations of KBLaM? How can users minimize the impact of KBLaM’s limitations when using the system?

When used with knowledge bases that are very different from the knowledge base it was trained on, KBLaM will give incomplete answers, and the answers can be reworded from the original value in the knowledge base or at times entirely incorrect. As a result, KBLaM is not currently intended for use as a complete system in a production setting, but is a research project that we are sharing.

### What operational factors and settings allow for effective and responsible use of KBLaM?

KBLaM with no knowledge base will perform the exact same as the base model. With a knowledge base, for effective use, one should make sure that the training dataset and the usecase have sufficiently similar knowledge bases

### How do I provide feedback on KBLaM?

Please add issues to this repository to provide feedback on KBLaM.
