import os
import torch
from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification
import re
def run():
    # 設定裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用裝置: {device}")

    # 載入模型和 tokenizer
    model_dir = "model/ner_model_medical_record"  # 修改成你模型存放的資料夾
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_dir)
    model = DebertaV2ForTokenClassification.from_pretrained(model_dir).to(device)

    # 必要的標籤對應
    id2label = model.config.id2label

    # 用來判斷是否是有效的醫療記錄號
    def is_valid_medical_record(record):
        # 保證符合數字或數字加.和英文的格式
        # 並排除單獨的年份或其他非醫療記錄的數字
        if re.match(r'^[0-9]+(\.[A-Za-z]+)?$', record):  # 數字或數字加.字母
            # 檢查是否是年份類的數字 (不包含在醫療記錄號碼中)
            if len(record) >= 4 and record.isdigit():
                return False  # 排除年份等純數字
            return True
        return False

    def extract_medical_number(text):
        # 找 medical record / medical record number，接著找 符合規範的編碼
        pattern = r"(medical record(?: number)?)(?:\s*(?:is|:|-|=)?\s*)([0-9]+(?:[-\.]?[0-9A-Za-z]+)*)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        numbers = []
        for match in matches:
            candidate = match[1]
            # 額外確保：開頭是數字，後面可以有 - . 字符和字母數字
            if re.match(r'^[0-9]+(?:[-\.]?[0-9A-Za-z]+)*$', candidate):
                numbers.append(candidate)
        return numbers




    # 預測單一文字
    def predict_entities(text):
        # 檢查是否包含 "medical record" 或 "medical record number"
        if "medical record" not in text.lower() and "medical record number" not in text.lower():
            return []  # 若沒有這些關鍵字，則返回空列表，不進行推理

        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        encoding = {k: v.to(device) for k, v in encoding.items()}  # encoding也搬到device
        with torch.no_grad():
            outputs = model(**encoding)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

        # 取回 offset_mapping (offset_mapping 只能重新encode，因為它不包含在原本的 tensor中)
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

        # 過濾掉不符合條件的 MEDICAL_RECORD_NUMBER
        filtered_entities = []
        for ent in entities:
            if ent["type"] == "MEDICAL_RECORD_NUMBER" and is_valid_medical_record(ent["text"]):
                filtered_entities.append(ent)

        # 若找不到符合的數字，使用 "medical number" 字串後的數字
        # 如果 NER一個都沒抓到 ➔ 直接規則補
        if not filtered_entities:
            extract_numbers = extract_medical_number(text)
            for number in extract_numbers:
                filtered_entities.append({"type": "MEDICAL_RECORD_NUMBER", "text": number})

        return filtered_entities


    # 推理一批資料
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
                # 只取 MEDICAL_RECORD_NUMBER 類別
                if ent["type"] == "MEDICAL_RECORD_NUMBER":
                    # 輸出結果符合你的需求: id 類別 實體值
                    results.append(f"{sid}\t{ent['type']}\t{ent['text']}")

        # 輸出
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for res in results:
                print(res)
                f.write(res + "\n")

        print(f"✅ 推理完成，結果儲存到 {output_file}")     
    run_inference("ASR_code/text/Whisper_Validation.txt", "validation/inference_medical_record_output.txt")