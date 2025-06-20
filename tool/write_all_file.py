def run():
    #初始化inference_output.txt
    content = ""
    with open("validation/inference_output.txt","w",encoding="utf-8") as f:
        f.write(content)

    #取得所有SHI TYPE預測結果
    with open("validation/inference_age_output.txt","r",encoding="utf-8") as f:
        age_output = f.read()
    with open("validation/inference_date_output.txt","r",encoding="utf-8") as f:
        date_output = f.read()
    with open("validation/inference_doctor_output.txt","r",encoding="utf-8") as f:
        doctor_output = f.read()
    with open("validation/inference_medical_record_output.txt","r",encoding="utf-8") as f:
        medical_record_output = f.read()
    with open("validation/inference_id_number_output.txt","r",encoding="utf-8") as f:
        id_number_output = f.read()
    with open("validation/inference_zip_output.txt","r",encoding="utf-8") as f:
        zip_output = f.read()
    with open("validation/inference_country_output.txt","r",encoding="utf-8") as f:
        country_output = f.read()
    with open("validation/inference_profession_output.txt","r",encoding="utf-8") as f:
        profession_output = f.read()
    with open("validation/inference_time_output.txt","r",encoding="utf-8") as f:
        time_output = f.read()
    with open("validation/inference_duration_output.txt","r",encoding="utf-8") as f:
        duration_output = f.read()
    with open("validation/inference_location_output.txt","r",encoding="utf-8") as f:
        location_output = f.read()
    with open("validation/inference_set_output.txt","r",encoding="utf-8") as f:
        set_output = f.read()
    with open("validation/inference_name_output.txt","r",encoding="utf-8") as f:
        name_output = f.read()
    #寫入所有SHI TYPE預測結果
    with open("validation/inference_output.txt","a",encoding="utf-8") as f:
        f.write(age_output)
        f.write(date_output)
        f.write(doctor_output)
        f.write(medical_record_output)
        f.write(id_number_output)
        f.write(zip_output)
        f.write(country_output)
        f.write(profession_output)
        f.write(time_output)
        f.write(duration_output)
        f.write(location_output)
        f.write(set_output)
        f.write(name_output)


    #刪除所有SHI TYPE預測結果
    with open("validation/inference_age_output.txt","w",encoding="utf-8") as f:
        f.write(content)
    with open("validation/inference_date_output.txt","w",encoding="utf-8") as f:
        f.write(content)
    with open("validation/inference_doctor_output.txt","w",encoding="utf-8") as f:
        f.write(content)
    with open("validation/inference_medical_record_output.txt","w",encoding="utf-8") as f:
        f.write(content)
    with open("validation/inference_id_number_output.txt","w",encoding="utf-8") as f:
        f.write(content)
    with open("validation/inference_zip_output.txt","w",encoding="utf-8") as f:
        f.write(content)
    with open("validation/inference_country_output.txt","w",encoding="utf-8") as f:
        f.write(content)
    with open("validation/inference_profession_output.txt","w",encoding="utf-8") as f:
        f.write(content)
    with open("validation/inference_time_output.txt","w",encoding="utf-8") as f:
        f.write(content)
    with open("validation/inference_duration_output.txt","w",encoding="utf-8") as f:
        f.write(content)
    with open("validation/inference_location_output.txt","w",encoding="utf-8") as f:
        f.write(content)
    with open("validation/inference_set_output.txt","w",encoding="utf-8") as f:
        f.write(content)
    with open("validation/inference_name_output.txt","w",encoding="utf-8") as f:
        f.write(content)
    with open("validation/inference_output_filtered.txt","w",encoding="utf-8") as f:
        f.write(content)
if __name__ == "__main__":
    run()