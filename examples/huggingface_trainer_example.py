# reference: https://github.com/minpeter/krill
from typing import List, cast

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from trl import pack_dataset

from pytorch_optimizer import Muon


def preprocess_dataset(tokenizer):
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples['text'], padding=False, truncation=False)

        if tokenizer.eos_token_id is not None:
            for i in range(len(tokenized_inputs['input_ids'])):
                tokenized_inputs['input_ids'][i].append(tokenizer.eos_token_id)
                tokenized_inputs['attention_mask'][i].append(1)
                if 'token_type_ids' in tokenized_inputs:
                    tokenized_inputs['token_type_ids'][i].append(0)

        return tokenized_inputs

    ds = load_dataset('HAERAE-HUB/KOREAN-WEBTEXT', split='train[:100]')

    tokenized = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)

    lengths: List[int] = (
        cast(List[int], tokenized['input_ids'].map(len))
        if hasattr(tokenized['input_ids'], 'map')
        else [len(x) for x in tokenized['input_ids']]
    )

    selected = [i for i, length in enumerate(lengths) if length >= 100]
    tokenized = tokenized.select(selected)

    packed = pack_dataset(tokenized, seq_length=1024, strategy='wrapped', map_kwargs={'batch_size': len(tokenized)})

    if len(packed) > 0:
        last_len = len(packed[-1]['input_ids'])
        if last_len < 1024:
            packed = packed.select(list(range(len(packed) - 1)))

    return packed


def get_model_config(tokenizer):
    config = LlamaConfig(
        initializer_range=0.02,
        hidden_size=16,
        num_hidden_layers=2,
        intermediate_size=64,
        tie_word_embeddings=False,
        num_attention_heads=4,
        num_key_value_heads=4,
    )

    config.torch_dtype = torch.bfloat16
    config.vocab_size = len(tokenizer)
    config.max_position_embeddings = 1024
    config.use_cache = False
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.eos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.rope_theta = 10_000.0

    return config


def get_optimizer(model):
    muon_params = [
        p
        for name, p in model.named_parameters()
        if p.ndim >= 2 and 'embed_tokens' not in name and 'lm_head' not in name
    ]
    non_muon_params = [
        p
        for name, p in model.named_parameters()
        if not (p.ndim >= 2 and 'embed_tokens' not in name and 'lm_head' not in name)
    ]

    param_groups = [
        {'params': muon_params, 'lr': 1e-3, 'weight_decay': 1e-2, 'use_muon': True},
        {'params': non_muon_params, 'lr': 1e-3, 'weight_decay': 1e-2, 'use_muon': False},
    ]

    return Muon(param_groups)


def main():
    tokenizer = AutoTokenizer.from_pretrained('minpeter/webtext-tokenizer-32k')

    config = get_model_config(tokenizer)

    model = LlamaForCausalLM(config)
    model.to(torch.bfloat16).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    optimizer = get_optimizer(model)

    ds = preprocess_dataset(tokenizer).train_test_split(test_size=0.2, shuffle=True, seed=42)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir='./',
        do_train=True,
        logging_dir='./logs',
        run_name='hf_train_example',
        overwrite_output_dir=True,
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        warmup_ratio=0.05,
        weight_decay=0.0,
        lr_scheduler_type='cosine',
        learning_rate=1e-3,
        bf16=True,
        dataloader_num_workers=2,
        dataloader_prefetch_factor=2,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        use_liger_kernel=False,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )
    trainer.train()


if __name__ == '__main__':
    main()
