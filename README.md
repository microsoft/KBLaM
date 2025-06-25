# KBLaM - Knowledge Base Augmented Language Models [ICLR 2025]

This repo contains the official implementation of [KBLaM: Knowledge Base Augmented Language Models](https://arxiv.org/abs/2410.10450).

Authors: Xi Wang, Liana Mikaelyan, Taketomo Isazawa, Mathew Salvaris, James Hensman.

KBLaM is a new method for augmentating LLMs with external knowledge. 
Unlike Retrieval-Augmented Generation, KBLAM eliminates external
retrieval modules, and unlike in-context learning, its computational overhead scales linearly with KB size rather than quadratically.

## Supported Models

The following models from Hugging Face hub are currently supported:
 - [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
 - [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
 - [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
 - [microsoft/bitnet-b1.58-2B-4T-bf16](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16)

To add support for new model types, you will need to update the model processing scripts to incorporate an adapter similar to `llama_model.py` in `src/kblam/models`.

## Setting up

Create and activate a Conda environment:
```bash
conda create -n kblam python=3.10.0
conda activate kblam
```

Install the kblam package with 

```
pip install -e .
```

To use Llama models, you will need to generate a token from Hugging Face and use it to log in:

```
pip install huggingface_hub
huggingface-cli login
```

The experiments in the paper can be replicated by running the scripts in `./experiments`.


## Dataset Construction

The dataset construction process involves two main steps.

### Step 1: Constructing the Dataset (Optional)

This step involves creating a new synthetic dataset from scratch.

To construct a synthetic KB and question-answer pairs, use `dataset_generation/gen_synthetic_data.py`. 

The question-answer pairs are constructed in the form:

```
What is the description of {entity_name}?
The description of {entity_name} is {description}.
```

**Alternatively, you can skip this step** and use the pre-generated datasets provided in the `datasets/` directory (`synthetic.json` or `enron.json`).

### Step 2: Generating KB Embeddings (Required)

This step takes a dataset file (e.g., `synthetic.json`) and generates the knowledge base embeddings required for training. You have two options for this process: using a paid OpenAI service or a free, local model.

#### Option A: OpenAI Embeddings (Paid)

This option uses the `text-embedding-ada-002` model via an Azure OpenAI endpoint. It is generally faster but will incur costs.

```bash
python dataset_generation/generate_kb_embeddings.py --model_name "ada-embeddings" --dataset_path "./datasets/synthetic.json" --dataset_name "synthetic" --endpoint_url "YOUR_AZURE_OPENAI_ENDPOINT" --output_path "./datasets"
```
- `--dataset_path`: Path to the input dataset file (e.g., `datasets/synthetic.json`).
- `--dataset_name`: A prefix for the output embedding files. The script will generate files like `synthetic_OAI_embd_key.npy`.
- `--endpoint_url`: **Required.** Your Azure OpenAI endpoint URL.

#### Option B: Local Sentence Transformer (Free)

This option uses the `all-MiniLM-L6-v2` model, which runs on your local machine. It is free but may be slower, especially without a GPU.

```bash
python dataset_generation/generate_kb_embeddings.py --model_name "all-MiniLM-L6-v2" --dataset_path "./datasets/synthetic.json" --dataset_name "synthetic" --output_path "./datasets"
```
- `--dataset_path`: Path to the input dataset file (e.g., `datasets/synthetic.json`).
- `--dataset_name`: A prefix for the output embedding files. The script will generate files like `synthetic_all-MiniLM-L6-v2_embd_key.npy`.


## Training

To train a model, run the `train.py` script with the desired arguments. The `--llm_type` argument specifies the base model architecture, and the `--encoder_spec` argument must match the model used to generate your KB embeddings in Step 2.

Note in particular the `--use_cached_embed` argument. This should be set to prevent recomputation of embeddings, which can take significant time especially when using APIs such as OpenAI's text embeddings.More actions

There are a number of optional arguments in `train.py` that you may want to consult.

### LLaMA-3 Examples

**Training with OpenAI Embeddings:**
```bash
python experiments/train.py --llm_type llama3 --hf_model_spec meta-llama/Llama-3.2-1B-Instruct --hf_token YOUR_HF_TOKEN --dataset_dir ./datasets --train_dataset synthetic --N 120000 --B 10 --total_steps 601 --encoder_spec OAI --use_cached_embd --key_embd_src key --use_data_aug
```

**Training with Local Sentence Transformer Embeddings:**
```bash
python experiments/train.py --llm_type llama3 --hf_model_spec meta-llama/Llama-3.2-1B-Instruct --hf_token YOUR_HF_TOKEN --dataset_dir ./datasets --train_dataset synthetic --N 120000 --B 10 --total_steps 601 --encoder_spec all-MiniLM-L6-v2 --use_cached_embd --key_embd_src key --use_data_aug
```

### Phi-3 Examples

**Training with OpenAI Embeddings:**
```bash
python experiments/train.py --llm_type phi3 --hf_model_spec microsoft/Phi-3-mini-4k-instruct --dataset_dir ./datasets --train_dataset synthetic --N 120000 --B 10 --total_steps 601 --encoder_spec OAI --use_cached_embd --key_embd_src key --use_data_aug
```

**Training with Local Sentence Transformer Embeddings:**
```bash
python experiments/train.py --llm_type phi3 --hf_model_spec microsoft/Phi-3-mini-4k-instruct --dataset_dir ./datasets --train_dataset synthetic --N 120000 --B 10 --total_steps 601 --encoder_spec all-MiniLM-L6-v2 --use_cached_embd --key_embd_src key --use_data_aug
```

### BitNet Examples

**Training with OpenAI Embeddings:**
```bash
python experiments/train.py --llm_type bitnet --hf_model_spec microsoft/bitnet-b1.58-2B-4T-bf16 --dataset_dir ./datasets --train_dataset synthetic --N 120000 --B 10 --total_steps 601 --encoder_spec OAI --use_cached_embd --key_embd_src key --use_data_aug
```

**Training with Local Sentence Transformer Embeddings:**
```bash
python experiments/train.py --llm_type bitnet --hf_model_spec microsoft/bitnet-b1.58-2B-4T-bf16 --dataset_dir ./datasets --train_dataset synthetic --N 120000 --B 10 --total_steps 601 --encoder_spec all-MiniLM-L6-v2 --use_cached_embd --key_embd_src key --use_data_aug
```

## Evaluation

To evaluate a trained model, use the `eval.py` script. You can evaluate generation quality, accuracy, and refusal.

The `--model_dir` and `--encoder_dir` arguments should point to the checkpoint directories created during the training step. For example, if your training script saved a model to `output/my_bitnet_model_step_601` and an encoder to `output/my_bitnet_model_step_601_encoder`, you would use those paths.

The examples below show how to evaluate for generation quality. The command structure is similar for other evaluation types like `accuracy` and `refusal`.

### LLaMA-3 Example

**Note:** You must provide a Hugging Face token to evaluate LLaMA models.
```bash
python experiments/eval.py generation --llm_type llama3 --llm_base_dir meta-llama/Llama-3.2-1B-Instruct --model_dir path/to/your/llama3/checkpoint --encoder_dir path/to/your/llama3/encoder --dataset_dir ./datasets --test_dataset synthetic.json --kb_size 200 --hf_token YOUR_HF_TOKEN
```

### Phi-3 Example
```bash
python experiments/eval.py generation --llm_type phi3 --llm_base_dir microsoft/Phi-3-mini-4k-instruct  --model_dir path/to/your/phi3/checkpoint --encoder_dir path/to/your/phi3/encoder --dataset_dir ./datasets --test_dataset synthetic.json --kb_size 200
```

### BitNet Example
```bash
python experiments/eval.py generation --llm_type bitnet --llm_base_dir microsoft/bitnet-b1.58-2B-4T-bf16 --model_dir path/to/your/bitnet/checkpoint --encoder_dir path/to/your/bitnet/encoder --dataset_dir ./datasets --test_dataset synthetic.json --kb_size 200
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
