import os
import torch
from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification
import re

def run():
    # 設定裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用裝置: {device}")

    # 載入模型和 tokenizer
    model_dir = "model/ner_model_id_number"
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_dir)
    model = DebertaV2ForTokenClassification.from_pretrained(model_dir).to(device)

    # 標籤映射
    id2label = model.config.id2label

    # 格式驗證函數（保守過濾）
    def is_valid_id_number(val):
        if re.match(r'^[A-Za-z0-9\-\.]+$', val):
            if re.search(r'[A-Za-z]', val) and re.search(r'\d', val):
                return True
            if val.isdigit():
                if len(val) >= 6 and not (len(val) == 4 and 1900 <= int(val) <= 2100):
                    return True
        return False

    def extract_id_number(text):
        pattern = r"(lab number|ID number|episode number)(?:\s*(?:is|:|-|=)?\s*)((?:[A-Za-z0-9\-\.]+\s*,\s*)*[A-Za-z0-9\-\.]+)"
        matches = re.findall(pattern, text, flags=re.IGNORECASE)

        numbers = []
        for label, value_group in matches:
            values = [v.strip().strip(".") for v in value_group.split(",")]
            for val in values:
                # 必須同時含有數字與英文字母，或是純數字且非年份
                if re.search(r'\d', val) and re.search(r'[A-Za-z]', val):  # 含數字與英文字
                    numbers.append(val)
                elif val.isdigit() and len(val) >= 6 and not (len(val) == 4 and 1900 <= int(val) <= 2100):
                    numbers.append(val)
        return numbers


    # 預測實體
    def predict_entities(text):
        if not any(kw in text.lower() for kw in ["id number", "lab number", "episode number"]):
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

        # 模型預測過濾：只保留格式合法的 ID
        filtered_entities = []
        for ent in entities:
            if ent["type"] == "ID_NUMBER" and is_valid_id_number(ent["text"]):
                filtered_entities.append(ent)

        # 若模型沒偵測，則使用規則補充（不做格式驗證）
        if not filtered_entities:
            extract_numbers = extract_id_number(text)
            for number in extract_numbers:
                filtered_entities.append({"type": "ID_NUMBER", "text": number})

        return filtered_entities

    # 跑整批輸入
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
                if ent["type"] == "ID_NUMBER":
                    results.append(f"{sid}\t{ent['type']}\t{ent['text']}")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for res in results:
                print(res)
                f.write(res + "\n")

        print(f"✅ 推理完成，結果儲存到 {output_file}")

    # 執行
    run_inference("ASR_code/text/Whisper_Validation.txt", "validation/inference_id_number_output.txt")