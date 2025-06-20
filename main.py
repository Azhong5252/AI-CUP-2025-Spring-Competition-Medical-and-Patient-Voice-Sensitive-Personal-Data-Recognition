from inference_code import inference_age,inference_date,inference_doctor,inference_medical_record,inference_id_number,inference_zip,inference_country,inference_profession,inference_time,inference_duration,inference_location,inference_set,inference_name
from tool import clear_allfile,write_all_file,sort_result,delete_repeat,Add_time,tab,filter_five_columns
import time
class MedicalSHITYPE:

    def __init__(self):
        self.initial_all = [clear_allfile]
        self.inference_all = [inference_age,inference_date,inference_doctor,inference_medical_record,inference_id_number,inference_zip,inference_country,inference_profession,inference_time,inference_duration,inference_location,inference_set,inference_name]# #inference_code SHI TYPE預測
        self.tool__all_ = [write_all_file,sort_result,Add_time,delete_repeat,tab,filter_five_columns]# #tool使用

    def execute(self,task):
        try:
            task.run()
        except Exception as e:
            print(f"Error_{task}")
        else:
            print(f"Correct_{task}")

    def initial_file_all(self):
        for i in self.initial_all:
            self.execute(i)

    def inference_code_all(self):
        for code in self.inference_all:
            self.execute(code)

    def tool_all(self):
        for t in self.tool__all_:
            self.execute(t)

    def run(self):
        print("Execute Inference:")
        self.inference_code_all()
        print("Execute tool:")
        self.tool_all()

if __name__ == "__main__":
    program = MedicalSHITYPE()
    program.initial_file_all()
    program.run()