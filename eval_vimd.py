import os
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "D:/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache/models"

import argparse
import time
import torch
import evaluate
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import DataLoader

TARGET_SR = 16000

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Whisper on VIMD dataset (Parquet format)")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--processor_name", type=str, default=None, help="Path/name of the original base model for the processor (e.g., openai/whisper-tiny)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing VIMD parquet files")
    parser.add_argument("--split", type=str, default="all", choices=["test", "valid", "all"], help="Which data split to evaluate (test, valid, all)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--language", type=str, default="vi", help="Language code")
    parser.add_argument("--task", type=str, default="transcribe", help="Task")
    parser.add_argument("--report_to", type=str, default="none", help="Reporting integrations: wandb, none")
    parser.add_argument("--run_name", type=str, default="whisper-eval-vimd", help="Run name for experiment tracking")
    return parser.parse_args()


def compute_and_print_results(all_predictions, all_references, total_inference_time, total_samples,
                               wer_metric, cer_metric, interrupted=False):
    """Compute final/partial metrics and print them."""
    label = "PARTIAL RESULTS (interrupted)" if interrupted else "EVALUATION RESULTS"
    wer = wer_metric.compute(predictions=all_predictions, references=all_references)
    cer = cer_metric.compute(predictions=all_predictions, references=all_references)
    avg_inference_time = total_inference_time / total_samples if total_samples > 0 else 0.0

    print("\n" + "="*40)
    print(label)
    print("="*40)
    print(f"Dataset         : VIMD Parquet ({total_samples} samples evaluated)")
    print(f"Total Samples   : {total_samples}")
    print(f"WER             : {wer * 100:.2f}%")
    print(f"CER             : {cer * 100:.2f}%")
    print(f"Avg Inf. Time   : {avg_inference_time:.4f} seconds/sample")
    print("="*40)

    return wer, cer, avg_inference_time


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading processor from {args.processor_name or args.model_name_or_path} ...")
    print(f"Loading model from {args.model_name_or_path} ...")
    
    proc_name = args.processor_name if args.processor_name else args.model_name_or_path
    processor = WhisperProcessor.from_pretrained(proc_name, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path).to(device)
    model.eval()

    print(f"Loading VIMD '{args.split}' dataset from {args.dataset_dir} ...")
    
    import glob
    # Resolve the data files based on the requested split
    if args.split == "test":
        files = glob.glob(os.path.join(args.dataset_dir, "test-*.parquet"))
    elif args.split == "valid":
        files = glob.glob(os.path.join(args.dataset_dir, "valid-*.parquet"))
    else: # all
        files = glob.glob(os.path.join(args.dataset_dir, "test-*.parquet")) + \
                glob.glob(os.path.join(args.dataset_dir, "valid-*.parquet"))

    if not files:
        print(f"No parquet files found in {args.dataset_dir} for split '{args.split}'")
        return

    # Load dataset directly from parquet files
    dataset = load_dataset("parquet", data_files={"eval": files}, split="eval")
    
    # Do not auto-decode, extract bytes manually
    import io
    import soundfile as sf
    import torchaudio
    
    dataset = dataset.cast_column("audio", Audio(decode=False))
    
    # Preprocess function to extract mel features
    def prepare_dataset(batch):
        audio_bytes = batch["audio"]["bytes"]

        with io.BytesIO(audio_bytes) as f:
            array, sr = sf.read(f, dtype="float32")

        array = torch.from_numpy(array) 

        if array.ndim > 1:
            array = array.mean(dim=1)
            
        if sr != TARGET_SR:
            array = torchaudio.functional.resample(array, sr, TARGET_SR)

        batch["input_features"] = processor.feature_extractor(array, sampling_rate=TARGET_SR).input_features[0]
        batch["reference"] = batch["text"]
        return batch

    print("Extracting features (this might take a moment) ...")
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=2)
    
    print(f"Dataset prepared with {len(dataset)} samples.")
    if len(dataset) == 0:
        print("No samples found! Check dataset path.")
        return

    # Keep dataset format as PyTorch tensors for DataLoader
    dataset.set_format(type="torch", columns=["input_features", "reference"])

    def collate_fn(batch):
        input_features = [item["input_features"] for item in batch]
        references = [item["reference"] for item in batch]
        input_features = torch.stack(input_features)
        return {"input_features": input_features, "references": references}

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    all_references = []
    all_predictions = []
    
    total_inference_time = 0.0
    total_samples = 0

    # Initialize WandB early so we can log per-step metrics
    wandb_run = None
    if args.report_to.lower() == "wandb":
        import wandb
        wandb_run = wandb.init(project="whisper-finetune", name=args.run_name, config=vars(args))
        print("WandB initialized. Will log per-step metrics.")

    print("Starting evaluation... (Press Ctrl+C to stop early and compute partial results)")

    interrupted = False
    step = 0

    try:
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_features = batch["input_features"].to(device)
                references = batch["references"]
                
                start_time = time.time()
                predicted_ids = model.generate(input_features)
                end_time = time.time()
                
                batch_inference_time = end_time - start_time
                total_inference_time += batch_inference_time
                total_samples += len(references)
                
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
                
                # Normalize text for fair comparison
                transcription = [t.lower() for t in transcription]
                references_norm = [r.lower() for r in references]
                
                all_predictions.extend(transcription)
                all_references.extend(references_norm)

                # Compute per-step (cumulative) metrics
                step_wer = wer_metric.compute(predictions=all_predictions, references=all_references)
                step_cer = cer_metric.compute(predictions=all_predictions, references=all_references)
                step_avg_inf_time = total_inference_time / total_samples

                step += 1

                # Print step metrics to console
                print(f"\n[Step {step}] samples={total_samples} | "
                      f"WER={step_wer*100:.2f}% | CER={step_cer*100:.2f}% | "
                      f"Avg Inf. Time={step_avg_inf_time:.4f}s/sample")

                # Log per-step metrics to WandB
                if wandb_run is not None:
                    wandb_run.log({
                        "eval/step_wer": step_wer,
                        "eval/step_cer": step_cer,
                        "eval/step_avg_inference_time": step_avg_inf_time,
                        "eval/samples_processed": total_samples,
                    }, step=step)

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n[!] Interrupted by user. Computing results from samples evaluated so far...")

    # Compute and print final (or partial) results
    if total_samples == 0:
        print("No samples were evaluated.")
        if wandb_run is not None:
            wandb_run.finish()
        return

    wer, cer, avg_inference_time = compute_and_print_results(
        all_predictions, all_references,
        total_inference_time, total_samples,
        wer_metric, cer_metric,
        interrupted=interrupted
    )

    # Log final/summary metrics to WandB
    if wandb_run is not None:
        tag = "partial" if interrupted else "final"
        wandb_run.log({
            f"eval/{tag}_wer": wer,
            f"eval/{tag}_cer": cer,
            f"eval/{tag}_avg_inference_time": avg_inference_time,
            f"eval/{tag}_total_samples": total_samples,
        })
        print(f"Logged {'partial' if interrupted else 'final'} summary results to WandB.")
        wandb_run.finish()


if __name__ == "__main__":
    main()
