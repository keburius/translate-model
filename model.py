import torch
from transformers import AutoModelForSeq2SeqLM, MT5Tokenizer

# model path for server ---> /home/ubuntu/model/v2-ss-tr.pt
# model path for local  ---> /Users/keburius/Desktop/NLP/Models/v2-ss-tr.pt


class TranslationModel:
    def __init__(self, model_repo='google/mt5-base', model_path='/home/ubuntu/model/v3-ss-tr'):
        self.LANG_TOKEN_MAPPING = {
            'DescriptionEn': '<en>',
            'DescriptionGe': '<ka>',
            'DescriptionRu': '<ru>',
        }

        self.tokenizer = MT5Tokenizer.from_pretrained(model_repo)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_repo, config={'max_new_tokens': 256}).to(self.device)

        special_tokens_dict = {'additional_special_tokens': list(self.LANG_TOKEN_MAPPING.values())}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def encode_input_str(self, text, target_lang, seq_len):
        target_lang_token = self.LANG_TOKEN_MAPPING[target_lang]

        input_ids = self.tokenizer.encode_plus(
            target_lang_token + text,
            padding='max_length',
            truncation=True,
            max_length=seq_len,
            return_tensors='pt'
        )['input_ids']

        return input_ids[0]

    def translate_text(self, input_text, output_language):
        input_ids = self.encode_input_str(
            text=input_text,
            target_lang=output_language,
            seq_len=256,
        )
        input_ids = input_ids.unsqueeze(0).to(self.device)

        output_tokens = self.model.generate(input_ids, num_beams=10,
                                            early_stopping=True, max_length=256)

        predicted_translation = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        return predicted_translation
