from  transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import pandas as pd
import requests
import json

API_URL =  "https://huggingface.co/api/datasets/medalpaca/medical_meadow_medqa/parquet/default/train"
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")


def query():
    response = requests.get(API_URL)
    return response.json()


def translate_sentence(sentence, tknzr, mdl):
    inputs = tknzr(sentence, return_tensors="pt")
    translated_tokens = mdl.generate(
        **inputs, forced_bos_token_id=tknzr.lang_code_to_id["fra_Latn"], max_length=150
    )
    result = tknzr.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return result


def translate_dataframe_cell(cell):
    return translate_sentence(cell, tokenizer, model)

def json2token(json_data):
    input_data = json_data["input"]
    input_to_token = input_data.tolist()
    instruction_data = json_data["instruction"]
    ouput_data = json_data["output"]
    return input_to_token

url_parquet = query()[0]
df = pd.read_parquet(url_parquet)
df_to_translate = df.head(1)
#df_to_translate.to_json('data_to_translate.json', orient="records")
sentence_input = json2token(df_to_translate)
tokens = tokenizer(sentence_input, return_attention_mask = False)
#for example in df_to_translate:
#     print(example)
#     example['input'] = translate_dataframe_cell(example['input'])
#     example['instruction'] = translate_dataframe_cell(example['instruction'])
#     example['output'] = translate_dataframe_cell(example['output'])
# df_to_translate.to_json('translated5.json', orient='records')
#
# df_translated = df_to_translate.applymap(translate_dataframe_cell)
# df_translated.to_json('translated_data.json', orient='records', lines=True)


batch_encoding = BatchEncoding(tokens)

vecteur = batch_encoding['input_ids'][0]

print(len(vecteur))