from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification
import torch
import re
import os

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = "model/ner_model_zip"
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_dir)
    model = DebertaV2ForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()

    input_path = "ASR_code/text/Whisper_Validation.txt"
    output_path = "validation/inference_zip_output.txt"

    label_map = model.config.id2label

    ZIP_HINTS = ["zip code", "postal code", "postal", "zip", "postcode"]

    def is_valid_zip(text, context):
        ctx = context.lower()
        text = text.strip()

        if not re.fullmatch(r"\d{4}", text):
            return False

        if text not in context:
            return False

        index = ctx.find(text)
        window_size = 20
        snippet = ctx[max(0, index - window_size): index + window_size] if index != -1 else ctx

        if not any(hint in snippet for hint in ZIP_HINTS):
            return False

        return True

    def extract_by_bio(text):
        encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=512)
        offset_mapping = encoding.pop("offset_mapping")[0]
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(**encoding)
        preds = torch.argmax(outputs.logits, dim=-1)[0].tolist()

        spans = []
        start = None
        for i, pid in enumerate(preds):
            label = label_map[pid]
            if label.startswith("B-ZIP"):
                start = i
            elif label == "O" and start is not None:
                spans.append((start, i-1))
                start = None
        if start is not None:
            spans.append((start, len(preds)-1))

        results = []
        for (s, e) in spans:
            start_char = offset_mapping[s][0]
            end_char = offset_mapping[e][1]
            span_text = text[start_char:end_char]
            results.append(span_text)
        return results

    def extract_by_regex(text):
        patterns = [
            r"\b(?:zip|postal|postcode)(?: code)?(?: for)?\s*(?:is|:)?\s*(\d{4})\b",
            r"\b\d{4}\b"
        ]
        results = []
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                if m.groups():
                    results.append(m.group(1))
                else:
                    results.append(m.group())
        return results

    with open(input_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    results = []
    seen = set()
    for line in lines:
        try:
            sid, sentence = line.split('\t', 1)
        except ValueError:
            continue

        for ent in extract_by_bio(sentence):
            val = ent.strip()
            if val and is_valid_zip(val, sentence):
                key = (sid, val)
                if key not in seen:
                    results.append(f"{sid}\tZIP\t{val}")
                    seen.add(key)

        for ent in extract_by_regex(sentence):
            val = ent.strip()
            if val and is_valid_zip(val, sentence):
                key = (sid, val)
                if key not in seen:
                    results.append(f"{sid}\tZIP\t{val}")
                    seen.add(key)
                    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in sorted(results):
            print(line)
            f.write(line + "\n")

    print(f"完成推理，已輸出至 {output_path}")