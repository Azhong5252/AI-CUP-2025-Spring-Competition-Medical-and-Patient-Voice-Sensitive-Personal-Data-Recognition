import os
import torch
import re
from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用裝置: {device}")

    model_dir = "model/ner_model_time"
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_dir)
    model = DebertaV2ForTokenClassification.from_pretrained(model_dir).to(device)
    id2label = model.config.id2label

    def is_valid_time(val):
        val = val.strip().lower()
        if re.search(r'\d{1,2}[.:：]\d{2}[.:：]\d{2}', val):
            return False
        patterns = [
            r'^\d{1,2}[:：.]\d{2}(\s*[ap]\.?m\.?)?$',
            r'^\d{1,2}\s*[ap]\.?m\.?$',
            r'^\d{1,2}-?ish$',
            r'^\d{1,2}\s*o[’\'`]clock$',
            r'^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(morning|afternoon|evening|night)$',
            r'^(today|tomorrow|yesterday|this|last|next|another|the)\s+(morning|afternoon|evening|night)$',
            r'^(early|late|middle of the|second)\s+(morning|afternoon|evening|night)$',
            r'^(noon|midnight|tonight|lunchtime|bedtime|dawn|dusk)$',
        ]
        return any(re.match(p, val) for p in patterns)

    def extract_time(text):
        patterns = [
            r'\b\d{1,2}[:：.]\d{2}(?![:：.\d])(?:\s*[ap]\.?m\.?)?\b',
            r'\b\d{1,2}\s*[ap]\.?m\.?\b',
            r'\b\d{1,2}-?ish\b',
            r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+o[’\'`]clock\b',
            r'\b(?:quarter|half|ten|twenty|five)\s+(?:past|to)\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b',
            r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(morning|afternoon|evening|night|bedtime|dawn|dusk|noon|midnight|lunchtime|tonight)\b',
            r'\b(?:last|this|next|another|today|tomorrow|yesterday|the)\s+(morning|afternoon|evening|night)\b',
            r'\b(?:early|late|middle of the|second)\s+(morning|afternoon|evening|night)\b',
            r'\b(?:noon|midnight|tonight|lunchtime|bedtime|dawn|dusk)\b',
        ]

        results = []
        for pat in patterns:
            matches = re.findall(pat, text, flags=re.IGNORECASE)
            for m in matches:
                text_match = " ".join(m).strip() if isinstance(m, tuple) else m.strip()
                if text_match:
                    results.append(text_match)
        return results

    def combine_time_phrases(text, entities):
        combined = []
        days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        prefixes = {"this", "last", "next", "another", "today", "tomorrow", "yesterday", "the"}
        modifiers = {"early", "late", "middle", "second"}

        for ent in entities:
            ent_text = ent["text"]
            start = text.find(ent_text)
            if start == -1:
                combined.append(ent)
                continue

            before_words = text[:start].strip().split()
            if not before_words:
                combined.append(ent)
                continue

            last_word = before_words[-1]
            second_last = before_words[-2] if len(before_words) >= 2 else ""

            if last_word in days:
                full = last_word + " " + ent_text
                combined.append({"type": "TIME", "text": full})
            elif last_word.lower() in prefixes:
                full = last_word + " " + ent_text
                combined.append({"type": "TIME", "text": full})
            elif second_last.lower() in modifiers and last_word.lower() == "of":
                if len(before_words) >= 3 and before_words[-3].lower() == "the":
                    full = " ".join(before_words[-3:] + [ent_text])
                    combined.append({"type": "TIME", "text": full})
                else:
                    combined.append(ent)
            else:
                combined.append(ent)
        return combined

    def predict_entities(text):
        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(**encoding)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

        offset_mapping = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=256)['offset_mapping']
        entities = []
        current_entity = None
        for idx, pred_id in enumerate(predictions):
            if idx >= len(offset_mapping):
                break
            start, end = offset_mapping[idx]
            if start == end:
                continue
            label = id2label.get(pred_id, "O")

            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {"type": label[2:], "text": text[start:end]}
            elif label.startswith("I-") and current_entity and label[2:] == current_entity["type"]:
                current_entity["text"] += text[start:end]
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        filtered = []
        for ent in entities:
            if ent["type"] == "TIME" and is_valid_time(ent["text"]):
                filtered.append(ent)

        if not filtered:
            extracted = extract_time(text)
            for t in extracted:
                filtered.append({"type": "TIME", "text": t})

        merged = combine_time_phrases(text, filtered)

        seen = set()
        unique = []
        for ent in merged:
            key = ent["text"].lower()
            if all(key not in other["text"].lower() or key == other["text"].lower() for other in merged if ent != other):
                if key not in seen:
                    unique.append(ent)
                    seen.add(key)

        return unique

    def run_inference(input_file, output_file):
        with open(input_file, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        results = []
        for line in lines:
            if "\t" not in line:
                continue
            sid, text = line.split("\t", 1)
            entities = predict_entities(text)
            for ent in entities:
                results.append(f"{sid}\t{ent['type']}\t{ent['text']}")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for res in results:
                print(res)
                f.write(res + "\n")

        print(f"✅ 推理完成，結果儲存到 {output_file}")

    run_inference("ASR_code/text/Whisper_Validation.txt", "validation/inference_time_output.txt")

if __name__ == "__main__":
    run()
