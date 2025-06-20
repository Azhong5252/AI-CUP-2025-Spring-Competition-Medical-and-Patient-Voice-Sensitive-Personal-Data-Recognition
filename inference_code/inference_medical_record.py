import os
import torch
from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification
import re
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用裝置: {device}")

    model_dir = "model/ner_model_medical_record"
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_dir)
    model = DebertaV2ForTokenClassification.from_pretrained(model_dir).to(device)

    id2label = model.config.id2label

    def is_valid_medical_record(record):
        if re.match(r'^[0-9]+(\.[A-Za-z]+)?$', record):  
            if len(record) >= 4 and record.isdigit():
                return False
            return True
        return False

    def extract_medical_number(text):
        pattern = r"(medical record(?: number)?)(?:\s*(?:is|:|-|=)?\s*)([0-9]+(?:[-\.]?[0-9A-Za-z]+)*)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        numbers = []
        for match in matches:
            candidate = match[1]
            if re.match(r'^[0-9]+(?:[-\.]?[0-9A-Za-z]+)*$', candidate):
                numbers.append(candidate)
        return numbers

    def predict_entities(text):
        if "medical record" not in text.lower() and "medical record number" not in text.lower():
            return []

        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        encoding = {k: v.to(device) for k, v in encoding.items()} 
        with torch.no_grad():
            outputs = model(**encoding)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

        encoding_cpu = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=256)
        offset_mapping = encoding_cpu['offset_mapping']

        entities = []
        current_entity = None
        current_text = ""
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
                entity_type = label[2:]
                current_text = text[start:end]
                current_entity = {"type": entity_type, "text": current_text}
            elif label.startswith("I-") and current_entity and label[2:] == current_entity["type"]:
                current_text += text[start:end]
                current_entity["text"] = current_text
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        filtered_entities = []
        for ent in entities:
            if ent["type"] == "MEDICAL_RECORD_NUMBER" and is_valid_medical_record(ent["text"]):
                filtered_entities.append(ent)

        if not filtered_entities:
            extract_numbers = extract_medical_number(text)
            for number in extract_numbers:
                filtered_entities.append({"type": "MEDICAL_RECORD_NUMBER", "text": number})

        return filtered_entities

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

                if ent["type"] == "MEDICAL_RECORD_NUMBER":

                    results.append(f"{sid}\t{ent['type']}\t{ent['text']}")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for res in results:
                print(res)
                f.write(res + "\n")

        print(f"推理完成，結果儲存到 {output_file}")     
    run_inference("ASR_code/text/Whisper_Validation.txt", "validation/inference_medical_record_output.txt")