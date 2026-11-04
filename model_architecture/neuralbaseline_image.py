import torch
import torch.nn as nn
from transformers import BartTokenizer, BartModel

# parameter setting
bart_emb_dim = 768
hidden_dim = 128
output_dim = 3


class NeuralBaselineImage(nn.Module):

    def __init__(self):
        super(NeuralBaselineImage, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.bart1 = BartModel.from_pretrained("facebook/bart-base").to(self.device)
        self.fc1 = nn.Linear(bart_emb_dim, hidden_dim).to(self.device)
        self.fc_last = nn.Linear(hidden_dim, output_dim).to(self.device)

    def forward(self, instances):
        #####################################
        #           Get Metadata            #
        #####################################

        sequences = []
        for instance in instances:

            # get everything from the json file
            image_caption = instance['metadata']['image_caption']
            sequences.append(image_caption)

        #####################################
        #             Encoding              #
        #####################################

        # use BART and maxpooling for encoding
        embedding1 = self.encode(sequences, self.bart1)

        #####################################
        #        Merging Modalities         #
        #####################################

        # individual representations
        embedding1 = self.fc1(embedding1)

        # make final prediction
        out = self.fc_last(embedding1)

        return out

    def encode(self, sequences, encoder):

        bart_inputs = self.bart_tokenizer(sequences, return_tensors="pt", truncation=True,
                                          padding='max_length', max_length=256).to(self.device)
        embedding = encoder(**bart_inputs)
        embedding = embedding.last_hidden_state
        sequence_length = embedding.shape[1]
        text_embedding = nn.MaxPool1d(kernel_size=sequence_length)(embedding.permute(0, 2, 1))
        embedding = text_embedding.squeeze(2)
        return embedding

