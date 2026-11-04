import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaModel

# parameter setting
roberta_emb_dim = 768
hidden_dim = 128
output_dim = 3


class TextOnly(nn.Module):

    def __init__(self, model):
        super(TextOnly, self).__init__()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.text_model = RobertaModel.from_pretrained("roberta-base").to(self.device)
        self.fc1 = nn.Linear(roberta_emb_dim, hidden_dim).to(self.device)

    def forward(self, instances):

        #####################################
        #           Get Metadata            #
        #####################################

        tweet_texts = [instance['metadata']['full_text'] for instance in instances]

        #####################################
        #             Encoding              #
        #####################################

        # use RoBERTa
        model_inputs = self.tokenizer(tweet_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        model_outputs = self.text_model(**model_inputs).pooler_output

        #####################################
        #          Classification           #
        #####################################

        # first step
        out = self.fc1(model_outputs)

        return out
