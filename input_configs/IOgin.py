import gin
from pathlib import Path
import shutil

@gin.configurable
class IOgin(object):
    def __init__(self, input_file, output_dir: str, name: str, csv_name: str):
        self.input_file = input_file
        self.output_dir = output_dir
        self.name = name
        self.csv_name = csv_name
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        print("Output folder is: ", self.output_dir)
        self.write_input_file()

    def write_input_file(self):
        shutil.copy(self.input_file, self.output_dir + "input_file.gin")
    

        