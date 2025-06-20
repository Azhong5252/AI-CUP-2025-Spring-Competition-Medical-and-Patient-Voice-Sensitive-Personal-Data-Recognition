from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification
import torch
import re
import os

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "model/ner_model_duration"
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_dir)
    model = DebertaV2ForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()

    input_path = "ASR_code/text/Whisper_Validation.txt"
    output_path = "validation/inference_duration_output.txt"
    label_map = model.config.id2label

    # 重新設計過的正則規則
    DURATION_REGEXES = [
        r"\b\d+(\.\d+)?\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?|sessions?)\b",
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
        r"\b(a\s+(couple|few)|several|some|many|past\s+(few|couple)|last\s+(few|couple|several))\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+and\s+a\s+half\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
        r"\b(a\s+half\s+(an\s+)?(second|minute|hour|day|week|month|year))\b",
        r"\b(throughout|the\s+whole|all|entire)\s+(day|week|month|year|night|weekend|semester|season)\b",
        r"\b(for|during|over|in|within|after|before|since)\s+(the\s+)?(last|past|next)?\s*\d*\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b"
    ]

    def is_potential_age_context(text, context):
        text = text.strip().lower()
        context = context.lower()
        age_keywords = ["years old", "year old", "age", "aged", "turning", "birthday", "born", "i am", "he is", "she is", "i’m", "i was"]
        if any(kw in context for kw in age_keywords):
            return True
        if re.search(r"\b\d+\s+years?\s+old\b", context):
            return True
        return False

    def is_valid_duration(text, context):
        text = text.strip().lower()
        context = context.lower()

        if len(text.split()) < 2 and text not in {
            "hour", "hours", "day", "days", "week", "weeks", "month", "months",
            "year", "years", "minute", "minutes", "moment", "while"
        }:
            return False

        false_positive = {
            "the", "that", "this", "so", "then", "know", "am", "have", "are", "will",
            "bit", "little", "one", "probably", "guess", "also", "oh", "i", "my", "his",
            "well", "was", "is", "great", "now", "like", "wanted", "quote", "what"
        }
        if text in false_positive:
            return False

        idioms = [
            "in the heat of the moment", "at the moment", "in this moment", "right now",
            "every moment", "one moment", "for the moment"
        ]
        if any(phrase in context for phrase in idioms):
            return False

        if is_potential_age_context(text, context):
            return False

        if text in {"moment", "while", "a while"}:
            index = context.find(text)
            window = context[max(0, index - 25): index + len(text) + 25]
            if not re.search(r"\b(for|last|past|during|throughout|over|within|in the)\b", window):
                return False

        index = context.find(text)
        if index != -1:
            window = context[max(0, index - 25): index + len(text) + 25]
            if re.search(r"\b(seconds?|minutes?|hours?|days?|weeks?|months?|years?|sessions?|duration|period|span|long|short|for)\b", window):
                return True

        return text in {
            "hour", "hours", "day", "days", "week", "weeks",
            "month", "months", "year", "years", "minute", "minutes",
            "moment", "while", "weekend", "evening", "night"
        }

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
            if label.startswith("B-DURATION"):
                if start is not None:
                    spans.append((start, i - 1))
                start = i
            elif label.startswith("I-DURATION") and start is not None:
                continue
            else:
                if start is not None:
                    spans.append((start, i - 1))
                    start = None
        if start is not None:
            spans.append((start, len(preds) - 1))

        results = []
        for s, e in spans:
            start_char = offset_mapping[s][0]
            end_char = offset_mapping[e][1]
            results.append(text[start_char:end_char])
        return results

    def extract_by_regex(text):
        results = []
        for pat in DURATION_REGEXES:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                results.append(m.group().strip())
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
            if is_valid_duration(ent, sentence):
                key = (sid, ent)
                if key not in seen:
                    results.append(f"{sid}\tDURATION\t{ent}")
                    seen.add(key)

        for ent in extract_by_regex(sentence):
            if is_valid_duration(ent, sentence):
                key = (sid, ent)
                if key not in seen:
                    results.append(f"{sid}\tDURATION\t{ent}")
                    seen.add(key)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in sorted(results):
            print(line)
            f.write(line + "\n")

    print(f"✅ 完成推理，已輸出至 {output_path}")

if __name__ == "__main__":
    run()
