import json
import random
import sys
sys.path.insert(0,r'./')
from tqdm.auto import tqdm

from configs import BaseConfig
from translator import DataParser


PARSER_NAME = "bluemoon"


class bluemoon(DataParser):
    def __init__(self, file_path: str, output_path: str, target_lang: str="ko",
                 max_example_per_thread=400, large_chunks_threshold=20000):
        super().__init__(file_path, output_path,
                         parser_name=PARSER_NAME,
                         do_translate=True,
                         target_lang=target_lang,
                         max_example_per_thread=max_example_per_thread,
                         large_chunks_threshold=large_chunks_threshold)
        self.max_ctxs = 5
        self.target_config = BaseConfig
        self.target_fields = ['source_text', 'target_text']

    def read(self) -> None:
        with open(self.file_path, encoding='utf-8') as jfile:
            json_data = json.load(jfile)
        self.data_read = json_data
        return None
        
def convert(self) -> None:
    data_converted = []
    for data in tqdm(self.data_read, desc="Converting data"):
        data_dict = {}
        data_dict['source_text'] = data['question']
        data_dict['target_text'] = data['answers'][0] if data['answers'] else None
        data_converted.append(data_dict)

    self.converted_data = data_converted
    return None


if __name__ == '__main__':
    bluemoon_parser = bluemoon(r"examples/ELI5/bluemoon.train.json",
                              r"examples/ELI5",
                              max_example_per_thread=400,
                              large_chunks_threshold=20000,
                              target_lang="ko")
    bluemoon_parser.read()
    bluemoon_parser.convert()
    bluemoon_parser.save()
