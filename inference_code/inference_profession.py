import torch
from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification
from typing import List, Tuple
import re
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "model/ner_model_profession"
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_path)
    model = DebertaV2ForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()

    id2label = model.config.id2label

    ALLOWED_PROFESSIONS = {
        "carpenter", "teacher", "lawyer", "engineer", "chef", "artist", "mechanic",
        "electrician", "plumber", "driver", "accountant", "scientist", "musician",
        "actor", "pilot", "architect", "dancer", "programmer", "designer", "barista",
        "photographer", "writer", "editor", "director", "producer", "banker",
        "consultant", "cashier", "receptionist", "waiter", "waitress", "bartender",
        "salesperson", "manager", "technician", "analyst", "researcher", "translator",
        "interpreter", "librarian", "firefighter", "police", "detective", "coach",
        "trainer", "real estate agent", "farmer", "gardener", "butcher", "tailor",
        "fashion designer", "chauffeur", "janitor", "custodian", "blacksmith",
        "welder", "truck driver", "delivery driver", "software engineer",
        "web developer", "data scientist", "system administrator", "animator", "entrepreneur","Archbishop","selling pottery","potter","trick cyclist",""

        "computer science", "mechanical engineering", "electrical engineering",
        "civil engineering", "biotechnology", "data science", "chemistry",
        "physics", "mathematics", "statistics", "psychology", "sociology",
        "political science", "economics", "business administration", "finance",
        "marketing", "literature", "philosophy", "linguistics", "history",
        "communications", "media studies", "environmental science", "geography",
        "anthropology", "criminology", "architecture", "education"
    }

    MEDICAL_JOBS = {
        "nurse", "doctor", "anesthetist", "surgeon", "therapist",
        "radiologist", "physician", "paramedic", "dentist", "midwife",
        "chiropractor", "psychiatrist", "optometrist", "pediatrician"
    }

    ACADEMIC_PATTERNS = [
        r"\bmajored in ([a-zA-Z\s]+)",
        r"\bstudied ([a-zA-Z\s]+)",
        r"\bdegree in ([a-zA-Z\s]+)",
        r"\bgraduated in ([a-zA-Z\s]+)"
    ]

    def is_allowed_profession(entity: str) -> bool:
        entity = entity.lower()
        return any(prof in entity for prof in ALLOWED_PROFESSIONS)

    def contains_medical_job(entity: str) -> bool:
        entity_words = re.findall(r"\b\w+\b", entity.lower())
        return any(word in MEDICAL_JOBS for word in entity_words)

    def augment_with_academic_context(text: str, entities: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        found_professions = {ent.lower() for ent, typ in entities if typ == "PROFESSION"}
        new_entities = []

        for pattern in ACADEMIC_PATTERNS:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                normalized = match.strip()
                if normalized in ALLOWED_PROFESSIONS and normalized not in found_professions:
                    new_entities.append((normalized, "PROFESSION"))
                    found_professions.add(normalized)

        return entities + new_entities

    def predict_shi_labels(text: str) -> List[Tuple[str, str]]:
        encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=256)
        offset_mapping = encoding["offset_mapping"].squeeze().tolist()
        encoding.pop("offset_mapping")

        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            output = model(**encoding)

        logits = output.logits
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()

        entities = []
        current_entity = ""
        current_type = ""

        for (start, end), pred_id in zip(offset_mapping, predictions):
            if start == end:
                continue
            label = id2label[pred_id]
            word = text[start:end]

            if label.startswith("B-"):
                if current_entity:
                    entities.append((current_entity, current_type))
                current_entity = word
                current_type = label[2:]
            elif label.startswith("I-") and current_type == label[2:]:
                current_entity += word
            else:
                if current_entity:
                    entities.append((current_entity, current_type))
                    current_entity = ""
                    current_type = ""

        if current_entity:
            entities.append((current_entity, current_type))

        filtered_entities = []
        for entity, ent_type in entities:
            if ent_type == "PROFESSION":
                if contains_medical_job(entity) or not is_allowed_profession(entity):
                    continue
            filtered_entities.append((entity, ent_type))

        filtered_entities = augment_with_academic_context(text, filtered_entities)

        return filtered_entities

    def process_validation_file(input_path: str, output_path: str):
        output_lines = []

        with open(input_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line in lines:
            parts = line.split("\t")
            if len(parts) == 2:
                sentence_id, sentence = parts
                entities = predict_shi_labels(sentence.strip())
                for entity, ent_type in entities:
                    output_lines.append(f"{sentence_id.strip()}\t{ent_type}\t{entity}\n")

        with open(output_path, "w", encoding="utf-8") as out_file:
            out_file.writelines(output_lines)
            print(output_lines)

    input_path = "ASR_code/text/Whisper_Validation.txt"
    output_path = "validation/inference_profession_output.txt"
    process_validation_file(input_path,output_path)
    print(f"推理完成，結果儲存到 {output_path}")
if __name__ == "__main__":
    run()