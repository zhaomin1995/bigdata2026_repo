import torch
import torch.nn as nn
from transformers import BartTokenizer, BartModel

# parameter setting
bart_emb_dim = 768
hidden_dim = 128
output_dim = 3


class NeuralBaselineImageLocDescAnnotQuestion(nn.Module):

    def __init__(self):
        super(NeuralBaselineImageLocDescAnnotQuestion, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.bart1 = BartModel.from_pretrained("facebook/bart-base").to(self.device)
        self.bart2 = BartModel.from_pretrained("facebook/bart-base").to(self.device)
        self.bart3 = BartModel.from_pretrained("facebook/bart-base").to(self.device)
        self.fc1 = nn.Linear(bart_emb_dim, hidden_dim).to(self.device)
        self.fc2 = nn.Linear(bart_emb_dim, hidden_dim).to(self.device)
        self.fc3 = nn.Linear(bart_emb_dim, hidden_dim).to(self.device)
        self.fc4 = nn.Linear(2 * bart_emb_dim, hidden_dim).to(self.device)
        self.fc5 = nn.Linear(2 * bart_emb_dim, hidden_dim).to(self.device)
        self.fc6 = nn.Linear(2 * bart_emb_dim, hidden_dim).to(self.device)
        self.fc7 = nn.Linear(3 * bart_emb_dim, hidden_dim).to(self.device)
        self.fc_last = nn.Linear(7 * hidden_dim, output_dim).to(self.device)

    def forward(self, instances):
        #####################################
        #           Get Metadata            #
        #####################################

        sequences1 = []
        sequences2 = []
        sequences3 = []
        for instance in instances:

            # get everything from the json file
            location_description = instance['metadata']['location_description']
            image_caption = instance['metadata']['image_caption']
            location_name = instance['metadata']['place']['full_name']
            annotation_question = f"How long will the author of the tweet " \
                                  f"continuously stay at {location_name} after publishing this tweet?"

            sequences1.append(annotation_question)
            sequences2.append(image_caption)
            sequences3.append(location_description)

        #####################################
        #             Encoding              #
        #####################################

        # use BART and maxpooling for encoding
        embedding1 = self.encode(sequences1, self.bart1)
        embedding2 = self.encode(sequences2, self.bart2)
        embedding3 = self.encode(sequences3, self.bart3)

        #####################################
        #        Merging Modalities         #
        #####################################

        # individual representations
        out1 = self.fc1(embedding1)
        out2 = self.fc2(embedding2)
        out3 = self.fc3(embedding3)

        # merge two modalities
        out4 = self.fc4(torch.cat((embedding1, embedding2), dim=1))
        out5 = self.fc5(torch.cat((embedding2, embedding3), dim=1))
        out6 = self.fc6(torch.cat((embedding1, embedding3), dim=1))

        # merge three modalities
        out7 = self.fc7(torch.cat((embedding1, embedding2, embedding3), dim=1))

        # make final prediction
        out = self.fc_last(torch.cat((out1, out2, out3, out4, out5, out6, out7), dim=1))

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
