from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification
import torch
import re
import os
from word2number import w2n

def run():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = "model/ner_model_age"
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_dir)
    model = DebertaV2ForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()

 
    input_path = "ASR_code/text/Whisper_Validation.txt"
    output_path = "validation/inference_age_output.txt"


    label_map = model.config.id2label


    def word_to_number(text):
        try:
            return str(w2n.word_to_num(text.lower()))
        except:
            return None


    def is_valid_age(text, context):
        ctx = context.lower()
        text = text.lower()

        if text not in ctx:
            return False


        id_like_patterns = [
            r"\b[0-9a-z]{6,}\b", 
            r"\b\d{5,}\b",       
        ]
        for pat in id_like_patterns:
            for m in re.finditer(pat, ctx):
                if text in m.group():
                    return False

        for m in re.finditer(r"(\d+\.){1,}\d+", ctx):
            if text in m.group():
                return False

        AGE_HINTS = ["year-old", "years old", "at age", "aged","at", "turning", "birthday", "celebrated", "was", "reached"]
        BIRTH_HINTS = ["born", "birthdate", "birthday", "date of birth"]
        MEDICAL_HINTS = ["hemoglobin", "blood", "mass", "lesion", "tumor", "size", "level", "count", "weight", "bpm", "pressure", "oxygen"]
        MONEY_HINTS = ["dollar", "$", "rent", "payment", "salary", "income", "cost", "amount", "bill"]

        index = ctx.find(text)
        window_size = 20
        snippet = ctx[max(0, index - window_size): index + window_size] if index != -1 else ctx

        if re.fullmatch(r"\d{2}s", text):
            return True

        if re.search(r"\b(he|she|they|i|you)'?s\s+\d{1,2}\b", snippet) or re.search(r"\b(he|He|She|she|they|They|I|i|you)\s+was\s+\d{1,2}\b", snippet):
            return True
        
        if any(hint in snippet for hint in MONEY_HINTS + MEDICAL_HINTS):
            return False

        if any(hint in snippet for hint in BIRTH_HINTS):
            return False

        if any(hint in snippet for hint in AGE_HINTS):
            pass
        else:
            return False

        if re.search(r"\b(he|she|they|i|you)\s+(just\s+)?turned\s+\d{1,2}\b", snippet):
            return True
        if re.search(r"\b(he|she|they|i|you)\s+turn(s|ed|ing)?\s+\d{1,2}\b", snippet):
            return True
        if re.search(r"\bturning\s+\d{1,2}\b", snippet):
            return True

        ENGLISH_NUMBERS = r"""
            one|two|three|four|five|six|seven|eight|nine|ten|
            eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|
            twenty( one| two| three| four| five| six| seven| eight| nine)?|
            thirty( one| two| three| four| five| six| seven| eight| nine)?|
            forty( one| two| three| four| five| six| seven| eight| nine)?|
            fifty( one| two| three| four| five| six| seven| eight| nine)?|
            sixty( one| two| three| four| five| six| seven| eight| nine)?|
            seventy( one| two| three| four| five| six| seven| eight| nine)?|
            eighty( one| two| three| four| five| six| seven| eight| nine)?|
            ninety( one| two| three| four| five| six| seven| eight| nine)?
        """

        if (
            re.search(r"\b(he|she|they|i|you)'?s\s+(" + ENGLISH_NUMBERS + r")\b", snippet, re.IGNORECASE | re.VERBOSE) or
            re.search(r"\b(he|she|they|i|you)\s+was\s+(" + ENGLISH_NUMBERS + r")\b", snippet, re.IGNORECASE | re.VERBOSE) or
            re.search(r"\b(" + ENGLISH_NUMBERS + r")\s*(year|years)( |-)?old\b", snippet, re.IGNORECASE | re.VERBOSE)
        ):
            return True

        if re.search(r"\bat\s+" + re.escape(text) + r"\b", ctx, re.IGNORECASE):
            try:
                if int(text) < 12:
                    return False
            except:
                num = word_to_number(text)
                if not num or int(num) < 12:
                    return False  
        if not re.search(r"\b\d{1,3}\s*(year|years)( |-)?old\b", ctx) and \
        not re.search(r"\b(" + ENGLISH_NUMBERS + r")\s*(year|years)( |-)?old\b", ctx, re.IGNORECASE | re.VERBOSE) and \
        not re.search(r"\b(he|she|they|i|you)\s+(just\s+)?(turned|turn|turns|turning|was|is|aged)\s+(" + ENGLISH_NUMBERS + r"|\d{1,3})\b", ctx, re.IGNORECASE | re.VERBOSE):
            return False

        return True

    def extract_by_bio(text):
        encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=256)
        offset_mapping = encoding.pop("offset_mapping")[0]
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(**encoding)
        preds = torch.argmax(outputs.logits, dim=-1)[0].tolist()

        spans = []
        start = None
        for i, pid in enumerate(preds):
            label = label_map[pid]
            if label.startswith("B-AGE"):
                start = i
            elif label == "O" and start is not None:
                spans.append((start, i-1))
                start = None
        if start is not None:
            spans.append((start, len(preds)-1))

        results = []
        for (s, e) in spans:
            if s >= len(offset_mapping) or e >= len(offset_mapping):
                continue  

            start_char = offset_mapping[s][0]
            end_char = offset_mapping[e][1]
            span_text = text[start_char:end_char]

            extra_text = text[end_char:end_char+20].lstrip()
            match = re.match(r"(year[-\s]old|years[-\s]old)", extra_text, re.IGNORECASE)
            if match:
                end_extra = match.end()
                span_text = text[start_char:end_char+end_extra]

            prev_text = text[max(0, start_char - 4):start_char]
            if re.search(r"\bat\s+$", prev_text, re.IGNORECASE):
                span_text = text[start_char - 3:end_char]

            results.append(span_text)        
        return results

    def extract_by_regex(text):
        patterns = [
            r"\b\d{1,3}\s+years old",
            r"\b\d{1,3}-year-old",
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[-\s]?(one|two|three|four|five|six|seven|eight|nine))?-year-old\b",
            r"\b\d{1,3}\s+year old\b",
            r"\bat the age of\s+\d{1,3}\b",
            r"\baged\s+\d{1,3}\b",
            r"\bturning\s+\d{1,3}\b",
            r"\b\d{1,3}(?:th)?\s+birthday\b",
            r"\breached\s+\d{1,3}\b",
            r"\bwas\s+\d{1,3}\b",
            r"\bcelebrated\s+.*?\b\d{1,3}(?:th)?\b",
            r"\b\d{2}s\b", 
            r"\b(?:he|she|they|i|you)'?s\s+\d{1,3}\b", 
            r"\b(?:he|she|they|i|you)\s+is\s+\d{1,3}\b",  
            r"\b(?:he|she|they|i|you)\s+was\s+\d{1,3}\b",
            r"\b(?:was|is|turned|turning|aged|reached)\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|[a-z]+(?:-[a-z]+)?)\b",
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|[a-z]+(?:-[a-z]+)?)\s+year(?:s)?(?:\s+old|-old)?\b",
            r"\bat\s+\d{1,3}\b",
            r"\bat\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[-\s]?(one|two|three|four|five|six|seven|eight|nine))?\b",
            r"\b(?:a|an)\s+\d{1,3}\s+year old\b"
        ]

        results = []
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                results.append(m.group())
        return results


    def normalize_age_phrase(phrase):
        phrase = phrase.lower().strip()


        if re.fullmatch(r"\d{2}s", phrase):
            return phrase

 
        m = re.search(r"\b\d{1,3}\b", phrase)
        if m:
            num = int(m.group(0))
            if 11 <= num <= 99:
                return str(num)

        try:
            word_match = re.search(r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[-\s]?(one|two|three|four|five|six|seven|eight|nine))?", phrase)
            if word_match:
                words = "-".join(filter(None, word_match.groups()))
                num = w2n.word_to_num(words)
                if 1 <= num <= 99:
                    return str(words)
        except:
            pass

        return None

    def convert_age_text_to_numbers(age_text):
        tens_keywords = {
            "ten", "twenty", "thirty", "forty", "fifty", "sixty",
            "seventy", "eighty", "ninety"
        }
        
        parts = age_text.split("-")

        if len(parts) == 1:
            return parts[0]

        second_part = parts[1].strip()
        second_words = second_part.split()
        
        if second_words[0] in tens_keywords:
            return second_part
        else:
            return second_words[0]

    def infer_age_from_sentence(text, model, tokenizer, label_map, device):
        candidates = extract_by_bio(text) + extract_by_regex(text)
        unique_candidates = list(set(candidates))
        filtered = [c for c in unique_candidates if is_valid_age(c, text)]
        normalized = [normalize_age_phrase(f) for f in filtered]
        final = sorted(set(n for n in normalized if n))
        return final
    
    input_path = "ASR_code/text/Whisper_Validation.txt"
    output_path = "validation/inference_age_output.txt"

    with open(input_path, "r", encoding="utf-8") as fin, \
        open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            parts = line.split(None, 1) 
            if len(parts) < 2:
                continue
            sid, text = parts[0], parts[1]

            ages = infer_age_from_sentence(text, model, tokenizer, label_map, device)
            for age_text in ages:
                age_num_text = convert_age_text_to_numbers(age_text)
                print(f"{sid}\tAGE\t{age_num_text}")
                fout.write(f"{sid}\tAGE\t{age_num_text}\n")
    print(f"完成推理，已輸出至 {output_path}")

if __name__ == "__main__":
    run()