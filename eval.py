import os
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "D:/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache/models"

import argparse
import time
import torch
import soundfile as sf
import torchaudio
import evaluate
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

TARGET_SR = 16000

class VivosDataset(Dataset):
    """Loads VIVOS from a local directory across one or more splits (e.g. train, test)."""
    def __init__(self, data_dir, processor, splits=("train", "test")):
        self.data_dir = data_dir
        self.processor = processor
        self.samples = []

        for split in splits:
            prompts_file = os.path.join(data_dir, split, "prompts.txt")
            waves_dir = os.path.join(data_dir, split, "waves")
            if not os.path.exists(prompts_file):
                print(f"[Warning] Prompts file not found for split '{split}': {prompts_file} — skipping.")
                continue
            print(f"  Loading local split '{split}' from {prompts_file} ...")
            with open(prompts_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        speaker_utt_id = parts[0]
                        transcript = parts[1]
                        speaker_id = speaker_utt_id.split("_")[0]
                        audio_path = os.path.join(waves_dir, speaker_id, f"{speaker_utt_id}.wav")
                        if os.path.exists(audio_path):
                            self.samples.append({
                                "audio_path": audio_path,
                                "text": transcript
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        array, sr = sf.read(sample["audio_path"], dtype="float32")
        array = torch.from_numpy(array)

        if array.ndim > 1:
            array = array.mean(dim=1)

        if sr != TARGET_SR:
            array = torchaudio.functional.resample(array, sr, TARGET_SR)

        input_features = self.processor.feature_extractor(array, sampling_rate=TARGET_SR, return_tensors="pt").input_features[0]

        return {
            "input_features": input_features,
            "reference": sample["text"]
        }

class VivosHFDataset(Dataset):
    """Loads VIVOS from the Hugging Face Hub across one or more splits (e.g. train, test)."""
    def __init__(self, dataset_name, processor, splits=("train", "test")):
        self.processor = processor
        all_splits = []
        for split in splits:
            print(f"  Loading HF split '{split}' from '{dataset_name}' ...")
            all_splits.append(load_dataset(dataset_name, split=split))
        # Concatenate all splits into one
        from datasets import concatenate_datasets
        self.samples = concatenate_datasets(all_splits)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio = sample["audio"]
        array = torch.tensor(audio["array"], dtype=torch.float32)
        sr = audio["sampling_rate"]

        if array.ndim > 1:
            array = array.mean(dim=1)

        if sr != TARGET_SR:
            array = torchaudio.functional.resample(array, sr, TARGET_SR)

        input_features = self.processor.feature_extractor(
            array, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features[0]

        return {
            "input_features": input_features,
            "reference": sample["sentence"],
        }


def collate_fn(batch):
    input_features = [item["input_features"] for item in batch]
    references = [item["reference"] for item in batch]
    input_features = torch.stack(input_features)
    return {"input_features": input_features, "references": references}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Whisper on VIVOS dataset")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--processor_name", type=str, default=None, help="Path/name of the original base model for the processor (e.g., openai/whisper-tiny) to avoid tokenizer load errors.")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to VIVOS dataset root. If absent or invalid, falls back to --hf_dataset_name.")
    parser.add_argument("--hf_dataset_name", type=str, default="AILAB-VNUHCM/vivos", help="HuggingFace dataset name used as fallback when dataset_dir is not available.")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"], help="Dataset splits to evaluate on (e.g. --splits train test).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--language", type=str, default="vi", help="Language code")
    parser.add_argument("--task", type=str, default="transcribe", help="Task")
    parser.add_argument("--report_to", type=str, default="none", help="Reporting integrations: wandb, none")
    parser.add_argument("--run_name", type=str, default="whisper-eval", help="Run name for experiment tracking")
    return parser.parse_args()


def compute_and_print_results(all_predictions, all_references, total_inference_time, total_samples,
                               wer_metric, cer_metric, interrupted=False, **kwargs):
    """Compute final/partial metrics and print them."""
    label = "PARTIAL RESULTS (interrupted)" if interrupted else "EVALUATION RESULTS"
    wer = wer_metric.compute(predictions=all_predictions, references=all_references)
    cer = cer_metric.compute(predictions=all_predictions, references=all_references)
    avg_inference_time = total_inference_time / total_samples if total_samples > 0 else 0.0

    print("\n" + "="*40)
    print(label)
    print("="*40)
    print(f"Dataset         : VIVOS ({', '.join(kwargs.get('splits', ['test']))} split(s))")
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

    splits = args.splits
    use_local = args.dataset_dir and os.path.isdir(args.dataset_dir)
    if use_local:
        print(f"Loading VIVOS dataset (splits: {splits}) from local path: {args.dataset_dir} ...")
        dataset = VivosDataset(args.dataset_dir, processor, splits=splits)
    else:
        if args.dataset_dir:
            print(f"[Warning] dataset_dir '{args.dataset_dir}' not found. Falling back to HF Hub.")
        print(f"Loading VIVOS dataset (splits: {splits}) from HF Hub: {args.hf_dataset_name} ...")
        dataset = VivosHFDataset(args.hf_dataset_name, processor, splits=splits)
    
    print(f"Found {len(dataset)} valid samples in the dataset.")
    if len(dataset) == 0:
        print("No samples found! Check dataset path.")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2)

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
        interrupted=interrupted,
        splits=splits
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
