from datasets import load_dataset, Audio, load_from_disk
import torch, torchaudio
import soundfile as sf
from typing import Dict
import io, os
import zipfile
from dataclasses import dataclass

TARGET_SR = 16000

class WhisperDataHandler:
    def __init__(self, config, processor):
        self.config = config
        self.processor = processor

    def prepare_dataset(self, batch):
        audio_bytes = batch["audio"]["bytes"]

        with io.BytesIO(audio_bytes) as f:
            array, sr = sf.read(f, dtype="float32")

        array = torch.from_numpy(array) 

        if array.ndim > 1:
            array = array.mean(dim=1)
            
        if sr != TARGET_SR:
            array = torchaudio.functional.resample(array, sr, TARGET_SR)

        # Feature Extractore
        batch["input_features"] = self.processor.feature_extractor(array, sampling_rate=TARGET_SR).input_features[0] #float32
        batch["labels"] = self.processor.tokenizer(batch["text"], max_length=448, truncation=True).input_ids #int64
        return batch

    def load_dataset(self, from_arrow=False):
        print(f"Load dataset: {self.config['dataset_name']} ...")

        if not from_arrow:
            raw_datasets = load_dataset(self.config['dataset_name'])
            def process_split(dataset):
                dataset = dataset.select_columns(["text", "audio"])
                dataset = dataset.cast_column("audio", Audio(decode=False))

                dataset = dataset.map(
                    self.prepare_dataset,
                    remove_columns=dataset.column_names,
                    batched=False,
                    num_proc=2
                )
                return dataset

            dataset_dict = {
                "train": process_split(raw_datasets["train"]),
                "valid": process_split(raw_datasets["valid"]),
                "test": process_split(raw_datasets["test"])
            }
        else:
            dataset_path = "/kaggle/input/datasets/ilewanducki/vimd-whisper-autotruncate/vimd-whisper-autotruncate"
            dataset_dict = {
                "train": load_from_disk(os.path.join(dataset_path, "ViMD_train_features")),
                "valid": load_from_disk(os.path.join(dataset_path, "ViMD_valid_features")),
                "test": load_from_disk(os.path.join(dataset_path, "ViMD_test_features"))
            }

        return dataset_dict

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: any

    def __call__(self, features) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Delete bos_token 'cause it's already implemented in forward step while fine-tune model 
        if (labels[:, 0].detach() == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch









