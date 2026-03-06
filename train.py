import yaml, os, argparse
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from src.data_loader import WhisperDataHandler, DataCollatorSpeechSeq2SeqWithPadding
from src.metrics import WERMetric


REQUIRED_KEYS = ["model_name", "output_dir", "language", "task"]


def load_config(path="configs/config.yaml"):
    """
    Load a YAML config file.
    - Warns (does not crash) when file is missing so pure-CLI usage still works.
    - Returns an empty dict when the file is absent or blank.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
            return cfg if isinstance(cfg, dict) else {}
    else:
        print(f"[Warning] Config file not found: '{path}'. Relying entirely on CLI arguments.")
        return {}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for speech recognition")

    # Config file
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to YAML config file (CLI args override config values)")

    # Model
    parser.add_argument("--model_name_or_path", type=str, help="HuggingFace model name or local path")
    parser.add_argument("--language", type=str, help="Language code, e.g. 'vi'")
    parser.add_argument("--task", type=str, help="Task: 'transcribe' or 'translate'")

    # Data
    parser.add_argument("--dataset_name", type=str, help="HuggingFace dataset name")
    parser.add_argument("--dataset_path", type=str, help="Local path to arrow dataset")

    # Actions
    parser.add_argument("--do_train", action="store_true", help="Run training")
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation only")

    # Output
    parser.add_argument("--output_dir", type=str, help="Directory to save checkpoints and final model")

    # Training duration
    parser.add_argument("--num_train_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, help="Max training steps (overrides epochs if > 0)")

    # Batch & accumulation
    parser.add_argument("--per_device_train_batch_size", type=int, help="Train batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, help="Eval batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps")

    # Optimizer
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, help="Warmup steps")
    parser.add_argument("--lr_scheduler_type", type=str, help="LR scheduler: linear, cosine, constant, etc.")

    # FIX: Use mutually exclusive group for fp16 so the intent is always unambiguous.
    # --fp16       → force enable  (cfg["fp16"] = True)
    # --no_fp16    → force disable (cfg["fp16"] = False)
    # neither flag → leave cfg["fp16"] as defined in YAML (or default False)
    fp16_group = parser.add_mutually_exclusive_group()
    fp16_group.add_argument("--fp16",    dest="fp16", action="store_true",  default=None,
                            help="Enable FP16 mixed precision")
    fp16_group.add_argument("--no_fp16", dest="fp16", action="store_false",
                            help="Disable FP16 mixed precision")

    # Eval & saving
    parser.add_argument("--eval_strategy", type=str, help="Evaluation strategy: 'steps' or 'epoch'")
    parser.add_argument("--save_steps", type=int, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, help="Log every N steps")
    parser.add_argument("--save_total_limit", type=int, help="Max number of checkpoints to keep")

    # Reporting
    parser.add_argument("--report_to", type=str, nargs="+", help="Reporting integrations: wandb, tensorboard, none")
    parser.add_argument("--run_name", type=str, help="Run name for experiment tracking")

    return parser.parse_args()


def build_config(args):
    """
    Merge strategy (highest priority last wins):
      1. Hardcoded defaults inside this function
      2. Values loaded from YAML config file
      3. Any CLI argument that was *explicitly* provided by the user

    A CLI arg is considered 'explicitly provided' when its value is not None
    (all optional args default to None so this check is reliable).
    """
    cfg = load_config(args.config)

    # Maps CLI dest-name  →  config key used throughout the rest of the script
    cli_to_cfg = {
        "model_name_or_path":           "model_name",
        "dataset_name":                 "dataset_name",
        "dataset_path":                 "dataset_path",
        "language":                     "language",
        "task":                         "task",
        "output_dir":                   "output_dir",
        "num_train_epochs":             "num_epochs",
        "max_steps":                    "max_steps",
        "per_device_train_batch_size":  "batch_size",
        "per_device_eval_batch_size":   "eval_batch_size",
        "gradient_accumulation_steps":  "gradient_accumulation_steps",
        "learning_rate":                "learning_rate",
        "weight_decay":                 "weight_decay",
        "warmup_steps":                 "warmup_steps",
        "lr_scheduler_type":            "lr_scheduler_type",
        "eval_strategy":                "eval_strategy",
        "save_steps":                   "save_steps",
        "eval_steps":                   "eval_steps",
        "logging_steps":                "logging_steps",
        "save_total_limit":             "save_total_limit",
        "report_to":                    "report_to",
        "run_name":                     "run_name",
    }

    # Override config with explicitly-provided CLI args (value is not None)
    for cli_key, cfg_key in cli_to_cfg.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            cfg[cfg_key] = val

    # FIX: args.fp16 is now None / True / False (mutually exclusive group).
    # Only write to cfg when the user explicitly passed a flag.
    if args.fp16 is not None:
        cfg["fp16"] = args.fp16

    # Handle do_train / do_eval flags
    cfg["do_train"] = args.do_train
    cfg["do_eval"]  = args.do_eval

    # FIX: Normalise report_to — always store as a list so downstream code
    # can iterate it without type-checking.
    report_to = cfg.get("report_to", ["wandb"])
    if isinstance(report_to, str):
        cfg["report_to"] = [report_to]

    # FIX: Validate that all required keys are present after merging.
    missing = [k for k in REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        raise ValueError(
            f"Missing required config value(s): {missing}. "
            "Provide them via the YAML config file or the corresponding CLI flags."
        )

    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)

    # If neither --do_train nor --do_eval is given, default to training
    do_train = cfg.get("do_train", False)
    do_eval  = cfg.get("do_eval", False)
    if not do_train and not do_eval:
        do_train = True

    print("Configuration loaded.")
    print(f"  Model     : {cfg.get('model_name')}")
    print(f"  Epochs    : {cfg.get('num_epochs', 3)}")
    print(f"  Batch size: {cfg.get('batch_size', 4)}")
    print(f"  LR        : {cfg.get('learning_rate', 1e-5)}")
    print(f"  Output    : {cfg.get('output_dir')}")
    print(f"  FP16      : {cfg.get('fp16', False)}")
    print(f"  do_train={do_train}, do_eval={do_eval}")

    # 1. Load processor & model
    processor = WhisperProcessor.from_pretrained(cfg['model_name'], language=cfg['language'], task=cfg['task'])
    model = WhisperForConditionalGeneration.from_pretrained(cfg['model_name'])

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # 2. Prepare Data
    data_handler = WhisperDataHandler(cfg, processor)
    full_dataset = data_handler.load_dataset(from_arrow=True)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric_computer = WERMetric(processor.tokenizer)

    # 3. Training Arguments
    training_args = Seq2SeqTrainingArguments(
            output_dir=cfg['output_dir'],

            # Duration
            num_train_epochs=cfg.get('num_epochs', 3),
            max_steps=cfg.get('max_steps', -1),

            # Batch & accumulation
            per_device_train_batch_size=cfg.get('batch_size', 4),
            per_device_eval_batch_size=cfg.get('eval_batch_size', 8),
            gradient_accumulation_steps=cfg.get('gradient_accumulation_steps', 1),

            # Optimizer
            learning_rate=float(cfg.get('learning_rate', 1e-5)),
            weight_decay=cfg.get('weight_decay', 0.0),
            warmup_steps=cfg.get('warmup_steps', 0),
            lr_scheduler_type=cfg.get('lr_scheduler_type', 'linear'),

            # Precision
            fp16=cfg.get('fp16', False),

            # Evaluation & saving
            eval_strategy=cfg.get('eval_strategy', 'steps'),
            save_steps=cfg.get('save_steps', 500),
            eval_steps=cfg.get('eval_steps', 500),
            logging_steps=cfg.get('logging_steps', 25),
            save_total_limit=cfg.get('save_total_limit', 3),
            load_best_model_at_end=cfg.get('load_best_model_at_end', True),
            metric_for_best_model=cfg.get('metric_for_best_model', 'wer'),
            greater_is_better=cfg.get('greater_is_better', False),

            # Generation
            predict_with_generate=True,
            generation_max_length=cfg.get('generation_max_length', 448),

            # Reporting
            report_to=cfg.get('report_to', ['wandb']),
            run_name=cfg.get('run_name', 'whisper-finetune-v1'),

            # Misc
            push_to_hub=cfg.get('push_to_hub', False),
            disable_tqdm=False,
            ddp_find_unused_parameters=False,
        )

    # 4. Initialize Trainer
    trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=full_dataset["train"],
            eval_dataset=full_dataset["test"],
            data_collator=data_collator,
            compute_metrics=metric_computer.compute_metrics,
            processing_class=processor.feature_extractor,
        )

    # 5. Auto-Resume Logic
    last_checkpoint = None
    if os.path.isdir(cfg['output_dir']):
            checkpoints = [d for d in os.listdir(cfg['output_dir']) if d.startswith("checkpoint-")]
            if len(checkpoints) > 0:
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                last_checkpoint = os.path.join(cfg['output_dir'], checkpoints[-1])
                print(f"Found checkpoint: {last_checkpoint}. Resuming training...")

    # 6. Run
    if do_train:
        trainer.train(resume_from_checkpoint=last_checkpoint)
        # Save Final Model
        trainer.save_model(os.path.join(cfg['output_dir'], "final_model"))
        processor.save_pretrained(os.path.join(cfg['output_dir'], "final_model"))

    if do_eval:
        results = trainer.evaluate()
        print("Evaluation results:")
        for key, value in results.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()