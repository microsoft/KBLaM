from transformers import AutoModelForCausalLM

from kblam.models.kblam_processor import KBLaMProcessor


def prune_str(S: str) -> str:
    S = S.replace('<|eot_id|>', ' ')
    S = S.replace('<|start_header_id|>assistant<|end_header_id|>', '')
    S = S.replace('<|start_header_id|>user<|end_header_id|>', '')
    S = S.replace('<|end_of_text|>', '')
    return S


def answer_question_old(model: AutoModelForCausalLM, tokenizer, Q: str, kb_config: dict, kb: dict = None) -> str:
    input_str = (
        '<|start_header_id|>user<|end_header_id|> ' + Q + '<|eot_id|>' + '<|start_header_id|>assistant<|end_header_id|>'
    )
    tokenizer_output = tokenizer(input_str, return_tensors='pt', padding=True).to('cuda')
    input_ids, attention_masks = tokenizer_output['input_ids'], tokenizer_output['attention_mask']
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        kb_kv=kb,
        max_new_tokens=70,
        tokenizer=tokenizer,
        output_attentions=True,
        kb_config=kb_config,
    ).squeeze()
    outputs = tokenizer.decode(outputs, skip_special_tokens=False)
    return prune_str(outputs)


def answer_question_new(model: AutoModelForCausalLM, tokenizer, Q: str, kb: dict = None) -> str:
    input_str = (
        '<|start_header_id|>user<|end_header_id|> ' + Q + '<|eot_id|>' + '<|start_header_id|>assistant<|end_header_id|>'
    )
    tokenizer_output = tokenizer(input_str, return_tensors='pt', padding=True).to('cuda')
    input_ids, attention_masks = tokenizer_output['input_ids'], tokenizer_output['attention_mask']
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        kb_kvs=kb,
        max_new_tokens=70,
        tokenizer=tokenizer,
        output_attentions=True,
    ).squeeze()
    outputs = tokenizer.decode(outputs, skip_special_tokens=False)
    return prune_str(outputs)


def answer_question(model: AutoModelForCausalLM, processor: KBLaMProcessor, Q: str, kb: list[tuple]) -> str:
    inputs = processor(knowledge_base=kb, text=Q)
    outputs = model.generate(
        **inputs,
        max_new_tokens=70,
        output_attentions=True,
    ).squeeze()
    outputs = processor.decode(outputs)
    return prune_str(outputs)
