import os
import re

def run():
    country_list = [
        "Afghanistan", "Albania", "Algeria", "American Samoa", "Andorra", "Angola", "Anguilla", "Antarctica",
        "Antigua and Barbuda", "Argentina", "Armenia", "Aruba", "Australia", "Austria", "Azerbaijan", "Bahamas",
        "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bermuda", "Bhutan", "Bolivia",
        "Bosnia and Herzegovina", "Botswana", "Brazil", "British Indian Ocean Territory", "Brunei Darussalam", "Bulgaria",
        "Burkina Faso", "Burundi", "Cabo Verde", "Cayman Islands", "Central African Republic", "Chad", "Chile", "China",
        "Christmas Island", "Cocos (Keeling) Islands", "Colombia", "Comoros", "Congo", "Congo, Democratic Republic of the",
        "Cook Islands", "Costa Rica", "Côte d'Ivoire", "Croatia", "Cuba", "Curaçao", "Cyprus", "Czechia", "Denmark",
        "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia",
        "Eswatini", "Ethiopia", "Falkland Islands (Malvinas)", "Faroe Islands", "Fiji", "Finland", "France", "French Guiana",
        "French Polynesia", "French Southern Territories", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Gibraltar",
        "Greenland", "Grenada", "Guadeloupe", "Guam", "Guatemala", "Guernsey", "Guinea", "Guinea-Bissau", "Guyana", "Haiti",
        "Heard Island and McDonald Islands", "Holy See", "Honduras", "Hong Kong", "Hungary", "Iceland", "India", "Indonesia",
        "Iran, Islamic Republic of", "Iraq", "Ireland", "Israel", "Isle of Man", "Italy", "Jamaica", "Japan", "Jersey",
        "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Korea, Democratic People's Republic of", "Korea, Republic of", "Kuwait",
        "Kyrgyzstan", "Lao People's Democratic Republic", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein",
        "Lithuania", "Luxembourg", "Macao", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Martinique",
        "Mauritania", "Mauritius", "Mayotte", "Mexico", "Micronesia (Federated States of)", "Moldova (Republic of)", "Monaco",
        "Mongolia", "Montenegro", "Montserrat", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands",
        "New Caledonia", "New Zealand", "Nicaragua", "Niger", "Nigeria", "Niue", "Norfolk Island", "North Macedonia",
        "Northern Mariana Islands", "Norway", "Oman", "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru",
        "Philippines", "Pitcairn", "Poland", "Portugal", "Puerto Rico", "Qatar", "Réunion", "Romania", "Russian Federation",
        "Rwanda", "Saint Barthélemy", "Saint Helena, Ascension and Tristan da Cunha", "Saint Kitts and Nevis", "Saint Lucia",
        "Saint Martin (French part)", "Saint Pierre and Miquelon", "Saint Vincent and the Grenadines", "Samoa", "San Marino",
        "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Sint Maarten (Dutch part)",
        "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Georgia and the South Sandwich Islands", "South Sudan",
        "Spain", "Sri Lanka", "Sudan", "Svalbard and Jan Mayen", "Sweden", "Switzerland", "Syrian Arab Republic", "Taiwan, Province of China",
        "Tajikistan", "Tanzania, United Republic of", "Thailand", "Timor-Leste", "Togo", "Tokelau", "Tonga", "Trinidad and Tobago", "Tunisia",
        "Türkiye", "Turkmenistan", "Turks and Caicos Islands", "Tuvalu", "United Arab Emirates", "United Kingdom of Great Britain and Northern Ireland",
        "United States of America", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela (Bolivarian Republic of)", "Viet Nam", "Western Sahara", "Yemen","USA",
        "Zambia", "Zimbabwe"
    ]
    exclude_list = [
        "Australian Capital Territory", "New South Wales", "Northern Territory",
        "Queensland", "South Australia", "Tasmania", "Victoria", "Western Australia",
        "Norfolk Island", "Christmas Island", "Cocos (Keeling) Islands",
        "Heard Island and McDonald Islands", "Coral Sea Islands",

        "American Samoa", "Guam", "Northern Mariana Islands", "Puerto Rico",
        "U.S. Virgin Islands", "United States Virgin Islands", "Navassa Island",
        "Midway Islands", "Wake Island", "Johnston Atoll", "Baker Island",
        "Howland Island", "Jarvis Island", "Kingman Reef", "Palmyra Atoll",

        "Anguilla", "Bermuda", "British Virgin Islands", "Cayman Islands",
        "Falkland Islands", "Falkland Islands (Malvinas)", "Gibraltar",
        "Montserrat", "Pitcairn Islands", "Saint Helena", "Ascension Island",
        "Tristan da Cunha", "South Georgia and the South Sandwich Islands",
        "Turks and Caicos Islands", "British Indian Ocean Territory",

        "French Guiana", "Guadeloupe", "Martinique", "Mayotte", "Réunion",
        "Saint Barthélemy", "Saint Martin", "Saint Pierre and Miquelon",
        "French Polynesia", "New Caledonia", "Wallis and Futuna",
        "French Southern and Antarctic Lands", "Clipperton Island",

        "Aruba", "Curaçao", "Sint Maarten", "Bonaire", "Sint Eustatius", "Saba",

        "Faroe Islands", "Greenland",

        "Svalbard", "Jan Mayen", "Bouvet Island",

        "Cook Islands", "Niue", "Tokelau", "Ross Dependency",

        "Hong Kong", "Macau", "Macao",

        "Isle of Man", "Jersey", "Guernsey",

        "Western Sahara", "Kosovo", "Palestine", "Taiwan", "Abkhazia",
        "South Ossetia", "Northern Cyprus", "Transnistria", "Nagorno-Karabakh",
        "Somaliland", "Gaza Strip", "West Bank", "Kurdistan Region",
        "Rojava", "Puntland", "Gagauzia", "Adjara", "Nakhchivan"
    ]

    task1_path = 'ASR_code/text/Whisper_Validation.txt'
    task2_path = 'validation/inference_country_output.txt'

    output_lines = []

    with open(task1_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            id_, sentence = line.split('\t', 1)

            if any(exclude in sentence for exclude in exclude_list):
                continue 

            for country in country_list:
                pattern = r'\b' + re.escape(country) + r'\b'
                if re.search(pattern, sentence):
                    output_lines.append(f"{id_}\tCOUNTRY\t{country}")

    with open(task2_path, 'w', encoding='utf-8') as f:
        for out_line in output_lines:
            print(out_line)
            f.write(out_line + '\n')

    print(f"已完成，結果儲存於 {task2_path}")