from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification
import torch
import os
import re

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = "model/ner_model_set"
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_dir)
    model = DebertaV2ForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()

    input_path = "ASR_code/text/Whisper_Validation.txt"
    output_path = "validation/inference_set_output.txt"

    label_map = model.config.id2label

    # ✳️ 語意提示詞
    SET_HINTS = [
        "every", "each", "per", "daily", "weekly", "monthly", "nightly", "morning", "evening",
        "twice", "once", "as needed", "times a day", "a couple of times", "several times", "multiple times"
    ]

    # ✳️ 正規模式（完整比對 or 子句中出現）
    SET_PATTERNS = [
        r"every \d+ (hours?|days?|weeks?)",
        r"once (a|per) (day|week|month)",
        r"twice (a|per) (day|week|month)",
        r"\d+ times (a|per) (day|week|month)",
        r"\d+/day",
        r"1/day",
        r"(once|twice|thrice|three times|four times) (daily|weekly|monthly|per week|per day)",
        r"(every|each) (morning|evening|night|afternoon)",
        r"(as needed|prn)",
        r"(nightly|daily|weekly|monthly)"
    ]
    # ❌ 模糊否定規則（不是 SET 的常見誤判）
    SET_NEGATIVE_HINTS = [
        "this morning", "that morning", "in the morning", "this evening", "in the evening",
        "friday morning", "tomorrow", "tonight", "yesterday", "later", "earlier",
        "last week", "last month", "next week", "next month", "soon", "before", "after", "today",
        "at bedtime", "in the afternoon", "now", "later in the evening", "that same morning"
    ]

    # ❌ 不是 SET 的常見錯詞（錯標的 phrase）
    SET_NEGATIVE_PATTERNS = [
        r"\b(today|tomorrow|yesterday|tonight|now|soon|later)\b",
        r"\b(this|that|last|next)\s+(morning|evening|week|month|night|afternoon|friday)\b",
        r"\bin the (morning|evening|night|afternoon)\b",
        r"\b[a-z]+\s+morning\b",   # like "Friday morning"
        r"\b[a-z]+\s+evening\b",
        r"\b[a-z]+\s+night\b",
        r"\bat\s+like\s+\d+\s+(am|pm|in the morning|in the evening)\b",
        r"\ba person\b",
        r"\beverywhere\b",
        r"\bsuperficially\b"
    ]

    def is_valid_set(text, context):
        snippet = text.strip().lower()

        if len(snippet.split()) > 10:
            return False

        # ❌ 檢查是否命中明確非SET片段
        if snippet in SET_NEGATIVE_HINTS:
            return False
        for pattern in SET_NEGATIVE_PATTERNS:
            if re.fullmatch(pattern, snippet) or re.search(pattern, snippet):
                return False

        # ✅ 判斷是否為SET
        if any(hint in snippet for hint in SET_HINTS):
            return True
        for pattern in SET_PATTERNS:
            if re.fullmatch(pattern, snippet) or re.search(pattern, snippet):
                return True

        return False


    def extract_set_entities(text):
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
            if label == "B-SET":
                if start is not None:
                    spans.append((start, i - 1))
                start = i
            elif label == "O":
                if start is not None:
                    spans.append((start, i - 1))
                    start = None
        if start is not None:
            spans.append((start, len(preds) - 1))

        results = []
        for s, e in spans:
            start_char = offset_mapping[s][0]
            end_char = offset_mapping[e][1]
            span_text = text[start_char:end_char].strip()
            if is_valid_set(span_text, text):
                results.append(span_text)
        return results

    # 主流程
    with open(input_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    results = []
    seen = set()
    for line in lines:
        try:
            sid, sentence = line.split('\t', 1)
        except ValueError:
            continue

        for ent in extract_set_entities(sentence):
            key = (sid, ent)
            if key not in seen:
                results.append(f"{sid}\tSET\t{ent}")
                seen.add(key)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in sorted(results):
            print(line)
            f.write(line + "\n")

    print(f"✅ 完成推理，結果已儲存至 {output_path}")

if __name__ == "__main__":
    run()