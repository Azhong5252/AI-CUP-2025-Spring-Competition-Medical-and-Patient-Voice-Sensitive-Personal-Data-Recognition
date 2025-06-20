# ✅ 完整版：改良後的訓練腳本 with BIO 檢查、Offset 對齊、類別加權

import os
import json
import re
from datasets import Dataset
from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification, TrainingArguments, Trainer,TrainerCallback
import torch
import torch.nn as nn
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

ALL_SHI_TYPES = ["DOCTOR"]
LABELS = ["O"] + [f"{prefix}-{t}" for t in ALL_SHI_TYPES for prefix in ("B", "I")]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

class SaveEvalResultsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        os.makedirs("training_eval", exist_ok=True)
        eval_file = "training_eval/eval_result_doctor.json"
        epoch_number = int(state.epoch) if state.epoch is not None else 0
        eval_entry = {
            "epoch": epoch_number,
            **metrics  
        }

        if os.path.exists(eval_file):
            with open(eval_file, "r", encoding="utf-8") as f:
                all_results = json.load(f)
                if not isinstance(all_results, list):
                    all_results = [all_results]
        else:
            all_results = []

        all_results.append(eval_entry)

        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)

        print(f"[Callback] 儲存 eval 結果（epoch={epoch_number}）至 {eval_file}")


tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")

def validate_bio_sequence(bio_tags):
    for i, tag in enumerate(bio_tags):
        if tag.startswith("I-") and (i == 0 or bio_tags[i - 1][2:] != tag[2:] or not bio_tags[i - 1].startswith(("B-", "I-"))):
            print(f" BIO錯誤 at position {i}: {tag} ← {bio_tags[i - 1]}")

def char_level_bio_encoding(text, entities):
    labels = ["O"] * len(text)
    for start, end, label in entities:
        if end > len(text) or start >= end:
            continue
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            if i < len(labels):
                labels[i] = f"I-{label}"
    validate_bio_sequence(labels)
    return labels

def align_labels_with_offsets(char_labels, offset_mapping):
    labels = []
    for start, end in offset_mapping:
        if start == end:
            labels.append(-100)
        else:
            span_labels = set(char_labels[start:end])
            span_labels.discard("O")
            if span_labels:
                tag = sorted(span_labels)[0]
                labels.append(label2id.get(tag, label2id["O"]))
            else:
                labels.append(label2id["O"])
    return labels

def load_data_char_based(task1_path, task2_path):
    with open(task1_path, encoding='utf-8') as f:
        texts = dict(line.strip().split('\t', 1) for line in f if line.strip())

    annotations = {}
    with open(task2_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 5:
                print(f" 無法解析此行（預期5個項目）: {line}")
                continue  
            try:
                sid, label, start, end, _ = parts
                annotations.setdefault(sid, []).append((int(float(start)), int(float(end)), label))
            except ValueError as e:
                print(f" 解析錯誤行: {line} - 錯誤: {e}")
                continue  

    dataset = []
    for sid, text in texts.items():
        entity_list = annotations.get(sid, [])
        char_labels = char_level_bio_encoding(text, entity_list)

        encoded = tokenizer(text, return_offsets_mapping=True, padding="max_length", truncation=True, max_length=256)
        encoded["labels"] = align_labels_with_offsets(char_labels, encoded["offset_mapping"])
        encoded.pop("offset_mapping")
        dataset.append(encoded)

    return dataset


class WeightedNERTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs): 
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weights = torch.ones(len(label2id)).to(logits.device)
        weights[label2id["O"]] = 0.1

        loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, len(label2id)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_preds = [[id2label[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]
    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
        "accuracy": accuracy_score(true_labels, true_preds)
    }

train_data = load_data_char_based("train_data/task1_doctor.txt", "train_data/task2_doctor.txt")
dataset = Dataset.from_list(train_data).train_test_split(test_size=0.2, seed=42)

model = DebertaV2ForTokenClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

args = TrainingArguments(
    output_dir="./ner_model",
    evaluation_strategy="epoch", 
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir="./logs/log_doctor",
    save_total_limit=1,
    logging_steps=500
)

trainer = WeightedNERTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.add_callback(SaveEvalResultsCallback())


last_checkpoint = None
if os.path.isdir(args.output_dir):
    checkpoints = [os.path.join(args.output_dir, d) for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints)[-1]
        print(f"🌀 找到現有 checkpoint，可從這裡恢復訓練：{last_checkpoint}")
    else:
        print("🆕 沒有 checkpoint，將從頭開始訓練")
else:
    print(" 尚未建立輸出資料夾，將從頭開始訓練")


trainer.train(resume_from_checkpoint=last_checkpoint)

model.save_pretrained("model/ner_model_doctor")
tokenizer.save_pretrained("model/ner_model_doctor")

print("模型訓練與儲存完成！")