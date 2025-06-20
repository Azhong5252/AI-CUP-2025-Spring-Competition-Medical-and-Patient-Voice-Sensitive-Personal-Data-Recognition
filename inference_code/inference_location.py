from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification
import torch
import re
import os

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VALID_ORGANIZATIONS = {
        "Walmart", "Amazon", "State Grid", "Saudi Aramco", "Sinopec Group", "China National Petroleum", "Apple",
        "UnitedHealth Group", "Berkshire Hathaway", "CVS Health", "Toyota Motor", "Volkswagen", "Samsung Electronics",
        "ExxonMobil", "Shell", "BP", "McKesson", "Daimler", "Glencore", "Cargill", "Costco", "Alphabet Inc.",
        "Microsoft", "Ping An Insurance", "Industrial and Commercial Bank of China", "China Construction Bank",
        "Agricultural Bank of China", "Bank of America", "JPMorgan Chase", "Wells Fargo", "Citigroup", "HSBC",
        "BNP Paribas", "Deutsche Bank", "Barclays", "Morgan Stanley", "Goldman Sachs", "UBS", "Credit Suisse",
        "Société Générale", "Royal Bank of Canada", "Toronto-Dominion Bank", "Bank of Montreal", "Scotiabank",
        "National Australia Bank", "Commonwealth Bank", "Westpac", "ANZ", "Mitsubishi UFJ Financial Group",
        "Sumitomo Mitsui Financial Group", "Mizuho Financial Group", "Nomura Holdings", "Daiwa Securities Group",
        "Resona Holdings", "Shinsei Bank", "Aozora Bank", "Norinchukin Bank", "Japan Post Bank", "ING Group",
        "Rabobank", "ABN AMRO", "KBC Group", "Dexia", "UniCredit", "Intesa Sanpaolo", "Banca Monte dei Paschi di Siena",
        "Banco Santander", "BBVA", "CaixaBank", "Bankia", "Banco Sabadell", "Nordea", "Danske Bank", "SEB",
        "Swedbank", "Handelsbanken", "DNB", "OP Financial Group", "Sberbank", "VTB Bank", "Gazprombank", "Alfa-Bank",
        "Bank of Moscow", "Raiffeisen Bank International", "Erste Group Bank", "Bank Austria", "BAWAG P.S.K.",
        "Hypo Group Alpe Adria", "Volksbank", "Banque Populaire", "Caisse d'Epargne", "Crédit Agricole",
        "La Banque Postale", "Natixis", "Banque Fédérative du Crédit Mutuel", "Crédit Mutuel", "Crédit du Nord",
        "HSBC France", "Société Générale", "LCL", "Facebook", "Oracle", "IBM", "Intel", "Cisco Systems",
        "Hewlett Packard Enterprise", "Dell Technologies", "SAP", "Salesforce", "Adobe", "Tencent", "Alibaba",
        "Baidu", "Xiaomi", "Lenovo", "Huawei", "ZTE", "LG Electronics", "Sony", "Panasonic", "Toshiba", "Fujitsu",
        "NEC", "Hitachi", "Sharp", "Nikon", "Canon", "Seiko Epson", "Ricoh", "Kyocera", "Brother Industries",
        "Pioneer", "JVC Kenwood", "Olympus", "Casio", "Citizen", "Mitsubishi Electric", "NTT Data", "SoftBank",
        "Rakuten", "LINE Corporation", "Kakao", "Naver", "SK Telecom", "KT Corporation", "LG Uplus", "Telstra",
        "Optus", "Vodafone", "BT Group", "Deutsche Telekom", "Orange", "Telefónica", "Telecom Italia", "Swisscom",
        "Telenor", "Telia Company", "Elisa", "Proximus", "KPN", "Belgacom", "Eir", "O2", "Three UK", "Virgin Media",
        "Liberty Global", "Comcast", "Charter Communications", "Cox Communications", "Altice USA", "Dish Network",
        "DirecTV", "AT&T", "Verizon", "T-Mobile US", "Sprint Corporation", "CenturyLink", "Frontier Communications",
        "Windstream Holdings", "Level 3 Communications", "Zayo Group", "Cogent Communications", "GTT Communications",
        "Intelsat", "SES S.A.", "Eutelsat", "Inmarsat", "Iridium Communications", "Globalstar", "Thuraya",
        "Hughes Network Systems", "Viasat", "EchoStar","Boston Scientific","Sealed Air Corporation","Delta Airlines","Reno Clinic and Southwestern Area","West Wimmera Health Service","Wimmera Health Care Group",
        "Cherbourg Hospital Cancer Center","Hunter Area Pathology Department","Springwood Hospital","Calvary Public Hospital ACT","Lewy Pathology"
    }

    ambiguous_city_terms = {
        "downtown", "urban area", "metro area", "central city", "inner city",
        "city center", "business district", "shopping district", "commercial area"
    }

    model_dir = "model/ner_model_location"
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_dir)
    model = DebertaV2ForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()

    input_path = "ASR_code/text/Whisper_Validation.txt"
    output_path = "validation/inference_location_output.txt"

    ALL_SHI_TYPES = [
        "ROOM", "DEPARTMENT", "HOSPITAL", "STREET", "CITY",
        "DISTRICT", "COUNTY", "STATE",  "LOCATION-OTHER",
    ]
    LABELS = ["O"] + [f"{prefix}-{t}" for t in ALL_SHI_TYPES for prefix in ("B", "I")]
    id2label = {i: l for i, l in enumerate(LABELS)}

    VALID_LABELS = {"HOSPITAL", "DEPARTMENT", "ROOM", "STREET", "CITY", "DISTRICT", "STATE", "ORGANIZATION","LOCATION-OTHER"}

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
            label = id2label[pid]
            if label.startswith("B-"):
                start = i
            elif label == "O" and start is not None:
                spans.append((start, i - 1))
                start = None
        if start is not None:
            spans.append((start, len(preds) - 1))

        results = []
        for (s, e) in spans:
            start_char = offset_mapping[s][0]
            end_char = offset_mapping[e][1]
            span_text = text[start_char:end_char]
            entity_label = id2label[preds[s]][2:]  
            results.append((entity_label, span_text))
        return results

    def extract_by_regex(text):
        return []

    def is_valid_location(label, text):
        t = text.strip()
        l = label.upper()

        hospital_keywords = ["hospital", "clinic", "medical center", "health center"]
        department_keywords = [
            "radiology", "oncology", "icu", "emergency", "surgery", "pathology",
            "cardiology", "internal medicine", "orthopedics", "psychiatry", "neurology", "gastroenterology"
        ]
        street_keywords = ["street", "st.", "avenue", "ave", "road", "rd.", "boulevard", "blvd", "lane", "drive", "highway", "way"]
        state_abbreviations = {"NSW", "ACT", "QLD", "VIC", "WA", "SA", "TAS", "NT", "CA", "NY", "TX", "FL", "IL", "PA", "OH"}

        text_lc = t.lower()

        if not t:
            return False

        if l in {"CITY", "DISTRICT", "COUNTY", "STATE"}:
            if t.isdigit():
                return False

        if l == "HOSPITAL":
            if not any(k in text_lc for k in hospital_keywords):
                return False

        if l == "DEPARTMENT":
            if t.lower() == "department":
                return False 
            department_keywords = [
                "radiology", "oncology", "icu", "emergency", "surgery", "pathology",
                "cardiology", "internal medicine", "orthopedics", "psychiatry", "neurology",
                "gastroenterology", "clinic", "ward", "unit", "department", "diagnostic", "laboratory", "imaging"
            ]
            if not any(k in text_lc for k in department_keywords):
                return False

        if l == "ROOM":
            if not any(k in text_lc for k in ["room", "floor", "suite"]):
                return False
            if not any(char.isdigit() for char in t):
                return False 

        if l == "STREET":
            if len(t.split()) < 2:
                return False  
            suffixes = ["street", "avenue", "road", "drive", "lane", "boulevard", "way", "place", "court", "crescent"]
            if not any(t.lower().endswith(s) for s in suffixes):
                return False
            if t.lower() in {"the street", "my street", "this street"}:
                return False 

        if l == "CITY":
            if len(t) < 2:
                return False
            text_lc_clean = re.sub(r"[^\w\s]", "", text_lc) 
            if any(term in text_lc_clean for term in ambiguous_city_terms):
                l = "LOCATION-OTHER"  


        if l == "DISTRICT":
            if "district" not in text_lc:
                return False

        if l == "COUNTY":
            if "county" not in text_lc:
                return False

        if l == "STATE":
            if len(t) < 2 and t.upper() not in state_abbreviations:
                return False
            
        def normalize_org(text):
            return re.sub(r'\W+', '', text).lower()
        if l == "ORGANIZATION":
            return normalize_org(t) in {normalize_org(org) for org in VALID_ORGANIZATIONS}
        
        return True

    def extract_department_by_keyword(text):
        department_keywords = [
            "radiology", "oncology", "icu", "emergency", "surgery", "pathology",
            "cardiology", "internal medicine", "orthopedics", "psychiatry", "neurology",
            "gastroenterology", "clinic", "ward", "unit", "department", "diagnostic", "laboratory", "imaging"
        ]
        results = []
        for kw in department_keywords:
            pattern = re.compile(rf"\b(?:department of|unit|ward|clinic)?\s*{kw}\b", re.IGNORECASE)
            match = pattern.search(text)
            if match:
                results.append(("DEPARTMENT", match.group().strip()))
        return results

    def extract_street_by_keyword(text):
        street_suffixes = [
            "Street", "Avenue", "Road", "Drive", "Lane", "Boulevard", "Way", "Place", "Court", "Crescent"
        ]
        patterns = [
            r"(?:lives at|resides on|lives on|moved to|address is|located at)\s+([\w\s]+(?:%s))" % suffix
            for suffix in street_suffixes
        ]
        results = []
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                street = match.group(1).strip()
                results.append(("STREET", street))
        return results

    
    def extract_organization_by_keyword(text):
        def normalize(s):
            return re.sub(r'\W+', '', s).lower()

        norm_org_set = {normalize(o): o for o in VALID_ORGANIZATIONS}
        results = []

        words = text.split()
        max_org_len = max(len(org.split()) for org in VALID_ORGANIZATIONS)

        for size in range(1, max_org_len + 1):
            for i in range(len(words) - size + 1):
                span = words[i:i+size]
                candidate = " ".join(span)
                norm_candidate = normalize(candidate)
                if norm_candidate in norm_org_set:

                    pattern = re.compile(re.escape(norm_org_set[norm_candidate]), re.IGNORECASE)
                    match = pattern.search(text)
                    if match:
                        results.append(("ORGANIZATION", match.group()))
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

        found_org = False

        for label, ent in extract_by_bio(sentence):
            val = ent.strip()
            if val and is_valid_location(label, val):
                key = (sid, label, val)
                if key not in seen:
                    results.append(f"{sid}\t{label}\t{val}")
                    seen.add(key)
                if label == "ORGANIZATION":
                    found_org = True

        for label, ent in extract_by_regex(sentence): 
            val = ent.strip()
            if val and is_valid_location(label, val):
                key = (sid, label, val)
                if key not in seen:
                    results.append(f"{sid}\t{label}\t{val}")
                    seen.add(key)

        if not found_org:
            for label, ent in extract_organization_by_keyword(sentence):
                val = ent.strip()
                if val and is_valid_location(label, val):
                    key = (sid, label, val)
                    if key not in seen:
                        results.append(f"{sid}\t{label}\t{val}")
                        seen.add(key)

        for label, ent in extract_department_by_keyword(sentence):
            val = ent.strip()
            if val and is_valid_location(label, val):
                key = (sid, label, val)
                if key not in seen:
                    results.append(f"{sid}\t{label}\t{val}")
                    seen.add(key)

        for label, ent in extract_street_by_keyword(sentence):
            val = ent.strip()
            if val and is_valid_location(label, val):
                key = (sid, label, val)
                if key not in seen:
                    results.append(f"{sid}\t{label}\t{val}")
                    seen.add(key)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in sorted(results):
            print(line)
            f.write(line + "\n")

    print(f"完成推理，已輸出至 {output_path}")

if __name__ == "__main__":
    run()