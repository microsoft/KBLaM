from transformers import AutoModelForCausalLM, AutoTokenizer

from kblam.models.kblam_processor import EncoderArgs, KBLaMProcessor
from kblam.models.llama3_2_model import KblamLlamaForCausalLM


def load_model_and_processor(
    model_path: str, encoder_name: str, kb_layer_frequency: int, encoder_dir: str
) -> tuple[AutoModelForCausalLM, KBLaMProcessor]:
    model = KblamLlamaForCausalLM.from_pretrained(model_path).bfloat16()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    args = EncoderArgs(
        encoder_name=encoder_name,
        hidden_size=model.config.hidden_size,
        num_hidden_layers=model.config.num_hidden_layers,
        kb_layer_frequency=kb_layer_frequency,
        encoder_dir=encoder_dir,
    )

    processor = KBLaMProcessor(tokenizer, args)
    return model, processor


if __name__ == "__main__":
    model_path = "/home/lmikaelyan/KBLaM/llama1B/stage1_lr_0.0001KBTokenLayerFreq3UseExtendedQAUseOutlier1KBSizedynamicSepQueryHeadKeyFromkey_OAI_synthetic_data_llama3_epoch_4300"
    model = KblamLlamaForCausalLM.from_pretrained(model_path, local_files_only=True).bfloat16()
    #tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # new_model = KblamLlamaForCausalLM(config).bfloat16()
    # new_model.load_state_dict(old_model.state_dict())