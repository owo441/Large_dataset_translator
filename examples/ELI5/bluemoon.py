import os
import json
import sys
from tqdm.auto import tqdm

sys.path.insert(0,r'./')

from configs import BaseConfig
from translator import DataParser

PARSER_NAME = "bluemoon"

class bluemoon(DataParser):
    def __init__(self, file_path: str, output_dir: str, output_file: str, target_lang: str="ko",
                 max_example_per_thread=400, large_chunks_threshold=20000):
        super().__init__(file_path, output_dir,
                         parser_name=PARSER_NAME,
                         do_translate=True,
                         target_lang=target_lang,
                         max_example_per_thread=max_example_per_thread,
                         large_chunks_threshold=large_chunks_threshold)
        self.output_path = os.path.join(output_dir, output_file)
        self.max_ctxs = 5
        self.target_config = BaseConfig
        self.target_fields = ['question_text', 'orig_answer_texts']

    def read(self) -> None:
        with open(self.file_path, encoding='utf-8') as jfile:
            json_data = json.load(jfile)
        self.data_read = json_data
        return None

    def convert(self) -> None:
        data_converted = []
        for data in tqdm(self.data_read, desc="Converting data"):
            data_dict = {}
            data_dict['qas_id'] = data['id']
            data_dict['question_text'] = ""
            data_dict['orig_answer_texts'] = ""
            for conv in data['conversations']:
                if conv['from'] == 'human':
                    data_dict['question_text'] += conv['value'] + "\n\n"
                else:  # This will include all non-human speakers as 'orig_answer_texts'
                    data_dict['orig_answer_texts'] += conv['value'] + "\n\n"
            data_dict['answer_lengths'] = None
            data_converted.append(data_dict)
        self.converted_data = data_converted
        return None

    def save(self) -> None:
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.converted_data, f, ensure_ascii=False, indent=4)
        return None

if __name__ == '__main__':
    bluemoon_parser = bluemoon(file_path=r"examples/ELI5/bluemoon.train.json",
                               output_dir=r"examples/ELI5",
                               output_file="bluemoon_converted.json",
                               max_example_per_thread=10,
                               large_chunks_threshold=100,
                               target_lang="ko")
    bluemoon_parser.read()
    bluemoon_parser.convert()
    bluemoon_parser.translate_converted()
    bluemoon_parser.save()
