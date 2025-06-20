import os
import torch
from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification

def run():
    MODEL_DIR = "model/ner_model_name"
    TOKENIZER_DIR = "model/ner_model_name"
    INPUT_FILE = "ASR_code/text/Whisper_Validation.txt"
    OUTPUT_FILE = "validation/inference_name_output.txt"

    SHI_TYPES = ["PATIENT", "FAMILYNAME", "PERSONALNAME"]
    LABELS = ["O"] + [f"{prefix}-{t}" for t in SHI_TYPES for prefix in ("B", "I")]
    label2id = {label: i for i, label in enumerate(LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DebertaV2TokenizerFast.from_pretrained(TOKENIZER_DIR)
    model = DebertaV2ForTokenClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    data = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sid, text = line.strip().split("\t", 1)
            data.append({"sid": sid, "text": text})

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for item in data:
            sid = item["sid"]
            text = item["text"]

            encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=256)
            encoding = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoding.items()}

            with torch.no_grad():
                outputs = model(**{k: v for k, v in encoding.items() if k != "offset_mapping"})
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

            offset_mapping = encoding["offset_mapping"].squeeze().tolist() if isinstance(encoding["offset_mapping"], torch.Tensor) else encoding["offset_mapping"]
            entities = []
            current_label = None
            current_start = None
            current_end = None

            for idx, label_id in enumerate(predictions):
                if idx >= len(offset_mapping): continue
                start, end = offset_mapping[idx]
                if start == end: continue

                label = id2label.get(label_id, "O")

                if label.startswith("B-"):
                    if current_label:
                        entities.append((current_label, current_start, current_end))
                    current_label = label[2:]
                    current_start = start
                    current_end = end
                elif label.startswith("I-") and current_label == label[2:]:
                    current_end = end
                else:
                    if current_label:
                        entities.append((current_label, current_start, current_end))
                    current_label = None
                    current_start = None
                    current_end = None

            if current_label:
                entities.append((current_label, current_start, current_end))

            for label, start, end in entities:
                span_text = text[start:end].strip()
                if len(span_text) < 2 or not any(c.isalpha() for c in span_text):
                    continue
                if label in ["PATIENT", "PERSONALNAME", "FAMILYNAME"] and len(span_text) < 3:
                    continue

                out_f.write(f"{sid}\t{label}\t{span_text}\n")
                print(f"{sid}\t{label}\t{span_text}")

    print(f"推理完成，結果已儲存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    run()