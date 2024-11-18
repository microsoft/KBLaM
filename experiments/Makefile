.DEFAULT_GOAL := help # Sets default action to be help


define PRINT_HELP_PYSCRIPT # start of Python section
import re, sys

output = []
# Loop through the lines in this file
for line in sys.stdin:
    # if the line has a command and a comment start with
    #   two pound signs, add it to the output
    match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
    if match:
        target, help = match.groups()
        output.append("%-10s %s" % (target, help))
# Sort the output in alphanumeric order
output.sort()
# Print the help result
print('\n'.join(output))
endef
export PRINT_HELP_PYSCRIPT # End of python section


LLAMA_BASE_DIR='/home/t-wangx/llama_weights/llama3_8b_ins'
LLAMA_HF='/datadisk/data/gcrbackup/hf_models/Meta-Llama-3-8B'
PHI3_HF='/datadisk/data/gcrbackup/hf_models/phi3'
DATASET_DIR='/datadisk/data/gcrbackup/oai_embd'
CKPT_SAVE_DIR='/home/msalvaris/data/experiments/kblam/exp_v0.2'
LR=1e-4
TRAIN_KB_SIZE=0 # Randomly pick KB size during training time
OUTLIER_RATIO=1 # Ratio of no-answer question in the batch, -1 stands for no such samples
KB_LAYER_FREQ=3 # How frequent to inject kb tokens into the layers
MULTI_ENTITIES_NUM=2 # For questions that involve multiple entities, how many entities are involved.


help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

train: ## Train kb adapter
	python train.py --model_dir ${LLAMA_BASE_DIR} \
					   --dataset_dir ${DATASET_DIR} \
					   --model_save_dir ${CKPT_SAVE_DIR} \
					   --seed 1607 \
					   --dataset gpt_data \
					   --N 120000 \
					   --B 20 \
					   --steps 20001 \
					   --encoder_spec all-MiniLM-L6-v2 \
					   --use_cached_embd \
					   --key_embd_src key \
					   --use_data_aug \
					   --use_lr_decay \
					   --tune_llm_q \
					   --sep_query_head \
					   --lr ${LR} \
					   --kb_size ${TRAIN_KB_SIZE} \
					   --kb_token_layer_frequency ${KB_LAYER_FREQ} \
					   --multi_entities ${MULTI_ENTITIES_NUM} \
					   --use_extended_qa \
					   --outlier_ratio ${OUTLIER_RATIO} \
					   --gradient_accm_step 20

train-oai: ## Train kb adapter
	python train.py --model_dir ${LLAMA_HF} \
					   --dataset_dir ${DATASET_DIR} \
					   --model_save_dir ${CKPT_SAVE_DIR} \
					   --seed 1607 \
					   --dataset gpt_data \
					   --N 120000 \
					   --B 2 \
					   --steps 20001 \
					   --encoder_spec OAI \
					   --use_cached_embd \
					   --key_embd_src key \
					   --use_data_aug \
					   --use_lr_decay \
					   --tune_llm_q \
					   --sep_query_head \
					   --lr ${LR} \
					   --no-duplicate_true_kb \
					   --kb_size ${TRAIN_KB_SIZE} \
					   --kb_token_layer_frequency ${KB_LAYER_FREQ} \
					   --multi_entities ${MULTI_ENTITIES_NUM} \
					   --use_extended_qa \
					   --outlier_ratio ${OUTLIER_RATIO} \
					   --gradient_accm_step 20




train-phi3-oai: ## Train kb adapter
	python train.py --model_dir ${PHI3_HF} \
					--dataset_dir ${DATASET_DIR} \
					--model_save_dir ${CKPT_SAVE_DIR} \
					--seed 1607 \
					--train_dataset synthetic_data \
					--N 120000 \
					--B 32 \
					--steps 20001 \
					--encoder_spec OAI \
					--use_cached_embd \
					--key_embd_src key \
					--use_data_aug \
					--use_lr_decay \
					--tune_llm_q \
					--sep_query_head \
					--lr ${LR} \
					--no-duplicate_true_kb \
					--kb_size ${TRAIN_KB_SIZE} \
					--kb_token_layer_frequency ${KB_LAYER_FREQ} \
					--multi_entities ${MULTI_ENTITIES_NUM} \
					--use_extended_qa \
					--outlier_ratio ${OUTLIER_RATIO} \
					--gradient_accm_step 10 \
					--llm_type 'phi3' \
					--use_cuda

eval-phi3-oai: ## Eval kb adapter
	python eval.py --seed 1607 \
				--ckpt_idx 40000 \
				--N 1000 \
				--test_dataset gpt_data \
				--train_dataset gpt_data \
				--ckpt_idx 40000 \
				--llm_ckpt_idx 40000 \
				--use_oai_embd \
				--dataset_dir ${DATASET_DIR} \
				--ckpt_dir ${CKPT_SAVE_DIR} \
				--model_dir ${PHI3_HF} \
				--encoder_spec OAI \
				--sep_query_head \
				--exp_config_str '0#Yes' \
				--use-cuda



eval-gen-phi3-oai: ## Eval kb adapter
	python eval.py \
				generation \
				--seed 1607 \
				--test_dataset synthetic_data \
				--ckpt_idx 20000 \
				--llm_base_dir ${PHI3_HF} \
				--encoder_dir '/datadisk/data/gcrbackup/experiments/kblam/exp_v0.1/stage1_lr_0.0001KBTokenLayerFreq3UseExtendedQAMultiEntities2UseOutlier1NoDuplicateKBSizedynamicSepQueryHeadUseDataAugFineTuneQueryKeyFromkey_synthetic_data_OAI_step_20000' \
				--model_dir '/datadisk/data/gcrbackup/experiments/kblam/exp_v0.1/stage1_lr_0.0001KBTokenLayerFreq3UseExtendedQAMultiEntities2UseOutlier1NoDuplicateKBSizedynamicSepQueryHeadUseDataAugKeyFromkey_OAI_synthetic_data_phi3_step_20000' \
				--dataset_dir ${DATASET_DIR} \
				--encoder_spec oai \
				--use_precomputed_embd \
				--eval_mode 'kb' \
				--save_dir '/datadisk/data/gcrbackup/experiments/kblam/exp_v0.2_eval' \
				--kb_token_layer_frequency ${KB_LAYER_FREQ} \
				--llm_type "phi3" \
				--kb_size 100



eval-acc-phi3-oai: ## Eval kb adapter
	python eval.py accuracy \
		--seed 1607 \
		--dataset_dir ${DATASET_DIR} \
		--test_dataset synthetic_data \
		--llm_base_dir ${PHI3_HF} \
		--model_dir '/datadisk/data/gcrbackup/experiments/kblam/exp_v0.1/stage1_lr_0.0001KBTokenLayerFreq3UseExtendedQAMultiEntities2UseOutlier1NoDuplicateKBSizedynamicSepQueryHeadUseDataAugKeyFromkey_OAI_synthetic_data_phi3_step_20000' \
		--encoder_dir '/datadisk/data/gcrbackup/experiments/kblam/exp_v0.1/stage1_lr_0.0001KBTokenLayerFreq3UseExtendedQAMultiEntities2UseOutlier1NoDuplicateKBSizedynamicSepQueryHeadUseDataAugFineTuneQueryKeyFromkey_synthetic_data_OAI_step_20000' \
		--save_dir '/datadisk/data/gcrbackup/experiments/kblam/exp_v0.2_eval' \
		--kb_layer_frequency ${KB_LAYER_FREQ} \
		--ckpt_idx 20000 \
		--lr 0 \
		--kb_size 100 \
		--no-fancy_instruction \
		--encoder_spec oai \
		--llm_type "phi3" \
		--attn_save_dir "/datadisk/kblam_attention" \
		--log_save_dir "/datadisk/kblam_attention/acc_results" 


eval-ref-phi3-oai: ## Eval kb adapter
	python eval.py refusal \
				--seed 1607 \
				--dataset_dir ${DATASET_DIR} \
				--test_dataset synthetic_data \
				--llm_base_dir ${PHI3_HF} \
				--encoder_dir '/datadisk/data/gcrbackup/experiments/kblam/exp_v0.1/stage1_lr_0.0001KBTokenLayerFreq3UseExtendedQAMultiEntities2UseOutlier1NoDuplicateKBSizedynamicSepQueryHeadUseDataAugFineTuneQueryKeyFromkey_synthetic_data_OAI_step_20000' \
				--model_dir '/datadisk/data/gcrbackup/experiments/kblam/exp_v0.1/stage1_lr_0.0001KBTokenLayerFreq3UseExtendedQAMultiEntities2UseOutlier1NoDuplicateKBSizedynamicSepQueryHeadUseDataAugKeyFromkey_OAI_synthetic_data_phi3_step_20000' \
				--ckpt_idx 20000 \
				--encoder_spec oai \
				--use_precomputed_embd \
				--eval_mode 'kb' \
				--save_dir '/datadisk/data/gcrbackup/experiments/kblam/exp_v0.2_eval' \
				--kb_token_layer_frequency ${KB_LAYER_FREQ} \
				--llm_type "phi3" \
				--kb_size 100


eval-basic-phi3-oai: ## Eval kb adapter
	python eval.py \
				standard \
				--seed 1607 \
				--test_dataset synthetic_data \
				--ckpt_idx 20000 \
				--llm_base_dir ${PHI3_HF} \
				--encoder_dir '/datadisk/data/gcrbackup/experiments/kblam/exp_v0.1/stage1_lr_0.0001KBTokenLayerFreq3UseExtendedQAMultiEntities2UseOutlier1NoDuplicateKBSizedynamicSepQueryHeadUseDataAugFineTuneQueryKeyFromkey_synthetic_data_OAI_step_20000' \
				--model_dir '/datadisk/data/gcrbackup/experiments/kblam/exp_v0.1/stage1_lr_0.0001KBTokenLayerFreq3UseExtendedQAMultiEntities2UseOutlier1NoDuplicateKBSizedynamicSepQueryHeadUseDataAugKeyFromkey_OAI_synthetic_data_phi3_step_20000' \
				--dataset_dir ${DATASET_DIR} \
				--encoder_spec oai \
				--use_precomputed_embd \
				--eval_mode 'kb' \
				--save_dir '/datadisk/data/gcrbackup/experiments/kblam/exp_v0.2_eval' \
				--kb_token_layer_frequency ${KB_LAYER_FREQ} \
				--llm_type "phi3" \
				--kb_size 100 \
				--exp_config_str "myconfig"

.PHONY: train train-oai
