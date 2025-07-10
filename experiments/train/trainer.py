import logging
import numpy as np
import pathlib
import torch
import transformers
import wandb

from functools import partial
from itertools import chain
from typing import Dict, List

from accelerate import Accelerator
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from transformers import get_linear_schedule_with_warmup

from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.models.bitnet_model import KBLaMBitNetForCausalLM
from kblam.models.gemma3n_model import KblamGemma3nForConditionalGeneration
from kblam.models.gemma3n_config import Gemma3nConfig

from .retriever import KBRetriever
from .batching import get_batch
from .data_formatting import (
    _format_QA_phi3, _create_labels_for_phi3,
    _format_QA_llama, _create_labels_for_llama,
    _format_QA_bitnet, _create_labels_for_bitnet,
    _format_QA_gemma3n, _create_labels_for_gemma3n
)
from .params import (
    _get_phi3_query_head_parameters,
    _get_llama3_query_head_parameters,
    _get_bitnet_query_head_parameters,
    _get_gemma3n_query_head_parameters
)
from .config import get_step_config
from .ui import create_custom_progress_bar, console

class Trainer:
    """A class for training the knowledge base language model.

    This class encapsulates the entire training process, including setting up the
    model, optimizer, and scheduler, running the training loop, and saving checkpoints.
    It uses Hugging Face's Accelerate for distributed training.

    Attributes:
        accelerator (Accelerator): The Accelerator object for distributed training.
        logger (logging.Logger): The logger for training.
        tokenizer: The tokenizer for the model.
        sep_query_head (bool): Whether to use a separate query head.
        kb_token_layer_frequency (int): The frequency of KB token layers.
        num_steps (int): The total number of training steps.
        lr (float): The learning rate.
        max_seq_len (int | None): The maximum sequence length.
        llm_type (str): The type of the language model.
        model: The language model.
        device: The device for training.
        kbretriever (KBRetriever): The knowledge base retriever.
        kb_size (int | List[int]): The size of the knowledge base.
        use_lr_decay (bool): Whether to use learning rate decay.
        llm_savename (str): The name for saving the language model.
        output_path (pathlib.Path): The path to the output directory.
        scheduler: The learning rate scheduler.
        optim: The optimizer.
    """
    def __init__(
        self,
        llm_model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM | KBLaMBitNetForCausalLM | KblamGemma3nForConditionalGeneration,
        kbretriever: KBRetriever,
        tokenizer: transformers.PreTrainedTokenizer,
        kb_token_layer_frequency: int,
        num_steps: int,
        lr: float,
        device: torch.device | None,
        use_lr_decay: bool,
        kb_size: int | List[int],
        llm_savename: str,
        output_dir: str,
        llm_type: str,
        sep_query_head: bool = False,
        max_seq_len: int | None = None,
    ):
        """Initializes the Trainer.

        Args:
            llm_model (KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM | KBLaMBitNetForCausalLM | KblamGemma3nForConditionalGeneration): The language model.
            kbretriever (KBRetriever): The knowledge base retriever.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
            kb_token_layer_frequency (int): The frequency of KB token layers.
            num_steps (int): The total number of training steps.
            lr (float): The learning rate.
            device (torch.device | None): The device for training.
            use_lr_decay (bool): Whether to use learning rate decay.
            kb_size (int | List[int]): The size of the knowledge base.
            llm_savename (str): The name for saving the language model.
            output_dir (str): The output directory.
            llm_type (str): The type of the language model.
            sep_query_head (bool, optional): Whether to use a separate query head. Defaults to False.
            max_seq_len (int | None, optional): The maximum sequence length. Defaults to None.
        """
        self.accelerator = Accelerator()
        self.logger = logging.getLogger("training")
        self.tokenizer = tokenizer
        self.sep_query_head = sep_query_head
        self.kb_token_layer_frequency = kb_token_layer_frequency
        self.num_steps = num_steps
        self.lr = lr
        self.max_seq_len = max_seq_len
        self.llm_type = llm_type

        self.model = llm_model
        self.model.gradient_checkpointing_enable()

        self.device = device if device is not None else self.accelerator.device
        self.kbretriever = kbretriever
        self.kb_size = kb_size
        self.use_lr_decay = use_lr_decay
        self.llm_savename = llm_savename
        self.output_path = pathlib.Path(output_dir)

        if isinstance(llm_model, KBLaMPhi3ForCausalLM):  # Phi3
            self._get_batch = partial(get_batch, _format_QA_phi3, _create_labels_for_phi3)
            self._get_params = _get_phi3_query_head_parameters
        elif isinstance(llm_model, KblamLlamaForCausalLM):  # llama
            self._get_batch = partial(get_batch, _format_QA_llama, _create_labels_for_llama)
            self._get_params = _get_llama3_query_head_parameters
        elif isinstance(llm_model, KBLaMBitNetForCausalLM):
            self._get_batch = partial(get_batch, _format_QA_bitnet, _create_labels_for_bitnet)
            self._get_params = _get_bitnet_query_head_parameters
        elif isinstance(llm_model, KblamGemma3nForConditionalGeneration):
            self._get_batch = partial(get_batch, _format_QA_gemma3n, _create_labels_for_gemma3n)
            self._get_params = _get_gemma3n_query_head_parameters
        else:
            raise ValueError(f"{llm_model} not recognised")

        self.scheduler, self.optim = self.setup_scheduler_and_optim()

        self.model, self.optim, self._get_batch, self.kbretriever.encoder = self.accelerator.prepare(
            self.model, self.optim, self._get_batch, self.kbretriever.encoder
        )

    def setup_scheduler_and_optim(self):
        """Sets up the optimizer and learning rate scheduler.

        This function configures the optimizer (AdamW) and an optional linear
        learning rate scheduler with warmup. It freezes the language model's
        backbone and only sets the query head(s) and the encoder to be trainable.

        Returns:
            tuple: A tuple containing the scheduler and optimizer.
        """
        # --- Refactored Optimizer Setup for All Models ---
        # 1. Get trainable parameters from the model (KBLaM modules)
        model_params_to_train = []
        if hasattr(self.model, 'get_trainable_parameters'):
            model_params_to_train = list(self.model.get_trainable_parameters())
            for p in model_params_to_train:
                p.requires_grad = True

        # 2. Always train the encoder
        encoder_params = list(self.kbretriever.encoder.parameters())
        for p in encoder_params:
            p.requires_grad = True

        # 3. Combine trainable parameters
        params_to_train = list(chain(model_params_to_train, encoder_params))

        # 4. Log trainable parameters
        trainable_names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        trainable_names += [n for n, p in self.kbretriever.encoder.named_parameters() if p.requires_grad]
        self.logger.info(f"Trainable parameters: {trainable_names}")

        # 5. Construct optimizer
        optim = AdamW(params_to_train, lr=self.lr)
        self.logger.info("Optimizer created for all trainable parameters.")

        # --- Linear LR decay with warmup ---
        scheduler = None
        if self.use_lr_decay:
            num_warmup_steps = int(0.06 * self.num_steps)
            scheduler = get_linear_schedule_with_warmup(
                optim,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.num_steps,
            )
            self.logger.info(f"Using linear LR scheduler with {num_warmup_steps} warmup steps.")
        return scheduler, optim

    def train(
        self,
        training_set: List[Dict],
        batch_size,
        grad_accum_steps: int,
        outlier_num: int,
        use_data_aug: bool = False,
        multi_entities: bool = False,
        use_extended_qa: bool = False,
        save_period: int = 100,
        resumed_step: int = 0,
        kb_config: KBLaMConfig | Gemma3nConfig = None,
    ):
        """Runs the training loop.

        This function executes the main training loop for the specified number of steps.
        It handles batching, forward and backward passes, optimization, and logging.
        It also saves model checkpoints periodically.

        Args:
            training_set (List[Dict]): The training dataset.
            batch_size (int): The batch size.
            grad_accum_steps (int): The number of gradient accumulation steps.
            outlier_num (int): The number of steps to use outlier questions.
            use_data_aug (bool, optional): Whether to use data augmentation. Defaults to False.
            multi_entities (bool, optional): Whether to use multi-entity questions. Defaults to False.
            use_extended_qa (bool, optional): Whether to use extended Q&A pairs. Defaults to False.
            save_period (int, optional): The period for saving checkpoints. Defaults to 100.
            resumed_step (int, optional): The step to resume training from. Defaults to 0.
            kb_config (KBLaMConfig | Gemma3nConfig, optional): The configuration for the knowledge base. Defaults to None.
        """
        train_losses = []
        start_step = resumed_step
        loss_fct = CrossEntropyLoss(reduction="none")

        # Use Accelerator for all model types, including BitNet
        num_processes = self.accelerator.num_processes
        accum_steps_per_gpu = max(1, grad_accum_steps // num_processes)
        effective_batch_size = batch_size * grad_accum_steps
        if self.accelerator.is_main_process:
            self.logger.info("================= KBLaM Training Start =================")
            self.logger.info(f"Model type: {self.llm_type}")
            self.logger.info(f"Training with {num_processes} GPUs")
            self.logger.info(f"Total steps: {self.num_steps}")
            self.logger.info(f"Total accumulation steps: {grad_accum_steps}, Steps per GPU: {accum_steps_per_gpu}")
            self.logger.info(f"Batch size: {batch_size}")
            self.logger.info(f"Effective batch size: {effective_batch_size}")
            self.logger.info(f"Learning rate: {self.lr}")
            self.logger.info(f"Max sequence length: {self.max_seq_len}")
            self.logger.info(f"Trainable parameters: {[n for n, p in list(self.model.named_parameters()) + list(self.kbretriever.encoder.named_parameters()) if p.requires_grad]}")
        with create_custom_progress_bar(console=console, disable=not self.accelerator.is_main_process) as pbar:
            task = pbar.add_task("Training", total=self.num_steps, loss=100)
            for step in range(start_step, self.num_steps, 1):
                self.optim.zero_grad()
                losses = []
                process_rank = self.accelerator.process_index
                start_accum_step = process_rank * accum_steps_per_gpu
                end_accum_step = min(start_accum_step + accum_steps_per_gpu, grad_accum_steps)
                # High-level step log
                if self.accelerator.is_main_process:
                    self.logger.info(f"================= Step {step} =================")
                for a_step in range(start_accum_step, end_accum_step):
                    step_config = get_step_config(
                        a_step,
                        grad_accum_steps,
                        use_data_aug,
                        outlier_num,
                        multi_entities,
                        use_extended_qa,
                    )
                    input_ids, attention_masks, labels, batch_indices = self._get_batch(
                        training_set,
                        self.tokenizer,
                        self.device,
                        B=batch_size,
                        random_sample=True,
                        **step_config,
                    )
                    if a_step == 0:
                        # Log input/label shapes and label mask summary
                        if self.accelerator.is_main_process:
                            self.logger.info(f"Batch input_ids shape: {input_ids.shape}, labels shape: {labels.shape}")
                            num_masked = (labels == -100).sum().item()
                            num_total = labels.numel()
                            num_unmasked = num_total - num_masked
                            self.logger.info(f"Label mask: masked={num_masked}, unmasked={num_unmasked}, pct_masked={num_masked/num_total:.2%}")
                            if num_unmasked == 0:
                                self.logger.warning("All labels are masked (-100)! Model will not learn.")
                            elif num_unmasked < 5:
                                self.logger.warning(f"Very few unmasked labels: {num_unmasked}")
                        # Log pre-step param values (first 5 elements)
                        for name, param in self.model.named_parameters():
                            if param.requires_grad and "q_proj" in name:
                                vals = param.view(-1)[:5].detach().cpu()
                                if vals.dtype == torch.bfloat16:
                                    vals = vals.to(torch.float32)
                                vals = vals.numpy()
                                self.logger.debug(f"Param pre-step {name}: first5={vals}")
                                self.logger.debug(f"Param device: {param.device}, dtype: {param.dtype}")
                                break
                    if self.max_seq_len is not None:
                        input_ids = input_ids[:, : self.max_seq_len]
                        attention_masks = attention_masks[:, : self.max_seq_len]
                        labels = labels[:, : self.max_seq_len]
                        if a_step == 0 and self.accelerator.is_main_process:
                            self.logger.info(f"TRUNCATED INPUT IDs SHAPE: {input_ids.shape}")
                    kb_embedding = self.kbretriever.get_key_embeddings(
                        batch_indices, len(input_ids), step, self.kb_size
                    )
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_masks,
                        kb_kvs=kb_embedding,
                        output_attentions=True,
                        kb_config=kb_config,
                    )
                    logits = out["logits"]
                    if a_step == 0 and self.accelerator.is_main_process:
                        with torch.no_grad():
                            batch_index = 0
                            max_logits = logits.argmax(axis=2)
                            decoded_pred = self.tokenizer.decode(max_logits[batch_index, :-1])
                            sel_labels = labels[batch_index, :]
                            sel_labels_unmasked = sel_labels[sel_labels != -100]
                            decoded_gt = self.tokenizer.decode(sel_labels_unmasked)
                            self.logger.info(f"KB embedding shape: {kb_embedding[0].shape}")
                            self.logger.info(f"Sample GT: {decoded_gt}")
                            self.logger.info(f"Sample PRED: {decoded_pred}")
                            wandb.log({"kbsize": kb_embedding[0].shape[1]})
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    weights = (shift_labels > 0).sum(-1, keepdim=True).expand(-1, shift_labels.shape[1]).contiguous()
                    model_config = (
                        self.model.config
                        if not isinstance(self.model, DistributedDataParallel)
                        else self.model.module.config
                    )
                    shift_logits = shift_logits.view(-1, model_config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    weights = weights.view(-1)
                    shift_labels = shift_labels.to(shift_logits.device)
                    raw_loss = loss_fct(shift_logits, shift_labels)
                    weighted_loss = raw_loss * weights.max() / weights
                    loss = weighted_loss.mean()
                    self.accelerator.backward(loss)
                    # Log gradient norm for key param
                    if a_step == 0:
                        for name, param in self.model.named_parameters():
                            if param.requires_grad and "q_proj" in name:
                                if param.grad is not None:
                                    grad_norm = float(torch.linalg.norm(param.grad).item())
                                    self.logger.debug(f"Grad {name}: norm={grad_norm:.6f}")
                                else:
                                    self.logger.debug(f"Grad {name}: None")
                                break
                        encoder_param = getattr(self.kbretriever.encoder, "projector_k", None)
                        if encoder_param is not None and hasattr(encoder_param, "weight") and encoder_param.weight.grad is not None:
                            self.logger.debug(f"KBEncoder grad norm: {torch.linalg.norm(encoder_param.weight.grad).item():.6f}")
                    losses.append(loss.item())
                # Log post-step param values (first 5 elements)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and "q_proj" in name:
                        vals = param.view(-1)[:5].detach().cpu()
                        if vals.dtype == torch.bfloat16:
                            vals = vals.to(torch.float32)
                        vals = vals.numpy()
                        self.logger.debug(f"Param post-step {name}: first5={vals}")
                        self.logger.debug(f"Param device: {param.device}, dtype: {param.dtype}")
                        break
                self.optim.step()
                if self.use_lr_decay and self.scheduler is not None:
                    self.scheduler.step()
                if losses:
                    local_loss = torch.tensor(np.mean(losses), device=self.device)
                else:
                    local_loss = torch.tensor(0.0, device=self.device)
                all_losses = self.accelerator.gather(local_loss)
                valid_losses = all_losses[all_losses > 0]
                avg_loss = valid_losses.mean().item() if len(valid_losses) > 0 else 0.0
                if self.accelerator.is_main_process:
                    self.logger.info(f"step: {step}, loss: {avg_loss}")
                    wandb.log({'train_loss': np.mean(losses)})
                    train_losses.append(avg_loss)
                    pbar.update(task, advance=1, loss=avg_loss)
                if (step % save_period) == 0 and (step != start_step):
                    try:
                        self.logger.info(
                            f"Is main process: {self.accelerator.is_main_process}, GPU memory before save: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB"
                        )
                        torch.cuda.empty_cache()
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            self.logger.info("Saving checkpoint...")
                            self.logger.info("Making dirs...")
                            model_ckpt_name = self.output_path / f"{self.llm_savename}_step_{step}"
                            model_ckpt_name.mkdir(parents=True, exist_ok=True)
                            encoder_dir = self.output_path / f"{self.llm_savename}_step_{step}_encoder"
                            encoder_dir.mkdir(parents=True, exist_ok=True)
                            self.logger.info("Saving model...")
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            unwrapped_model.save_pretrained(
                                model_ckpt_name,
                                is_main_process=self.accelerator.is_main_process,
                                save_function=self.accelerator.save,
                                safe_serialization=False,
                            )
                            self.logger.info("Saving tokenizer...")
                            self.tokenizer.save_pretrained(model_ckpt_name)
                            self.logger.info("Saving encoder...")
                            encoder_ckpt_name = encoder_dir / "encoder.pt"
                            torch.save(self.kbretriever.encoder.state_dict(), encoder_ckpt_name)
                            self.logger.info("Saving config...")
                            config_path = model_ckpt_name / "kb_config_explicit.json"
                            with open(config_path, 'w') as f:
                                f.write(kb_config.to_json_string())
                    except Exception as e:
                        self.logger.error(f"Error saving checkpoint: {e}")
                        self.logger.error(f"Error details: {str(e)}")
                        raise e
        if self.accelerator.is_main_process:
            self.logger.info("================= KBLaM Training Complete =================")
