import torch
import torch.nn as nn
from transformers import BartTokenizer, BartModel

# parameter setting
bart_emb_dim = 768
hidden_dim = 128
output_dim = 3


class RecursiveAll4(nn.Module):

    def __init__(self, model):
        super(RecursiveAll4, self).__init__()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.bart = BartModel.from_pretrained("facebook/bart-base").to(self.device)
        self.fc1 = nn.Linear(bart_emb_dim, hidden_dim).to(self.device)
        self.fc2 = nn.Linear(bart_emb_dim, hidden_dim).to(self.device)
        self.fc3 = nn.Linear(bart_emb_dim, hidden_dim).to(self.device)
        self.fc4 = nn.Linear(bart_emb_dim, hidden_dim).to(self.device)
        self.fc5 = nn.Linear(2 * hidden_dim, hidden_dim).to(self.device)
        self.fc6 = nn.Linear(2 * hidden_dim, hidden_dim).to(self.device)
        self.fc_last = nn.Linear(2 * hidden_dim, output_dim).to(self.device)

    def forward(self, instances):
        #####################################
        #           Get Metadata            #
        #####################################

        sequences1 = []
        sequences2 = []
        sequences3 = []
        sequences4 = []
        for instance in instances:

            # get everything from the json file
            tweet_text = instance['metadata']['full_text']
            image_caption = instance['metadata']['image_caption']
            location_description = instance['metadata']['location_description']
            location_name = instance['metadata']['place']['full_name']
            annotation_question = f"How long will the author of the tweet " \
                                  f"continuously stay at {location_name} after publishing this tweet?"

            # the model deals with sequences1 & sequences2 firstly, then sequences3, then sequences4
            # based on the given model type, we change the ordering
            sequence1, sequence2, sequence3, sequence4 = self.order_input(
                self.model, tweet_text, location_description, image_caption, annotation_question
            )
            sequences1.append(sequence1)
            sequences2.append(sequence2)
            sequences3.append(sequence3)
            sequences4.append(sequence4)

        #####################################
        #             Encoding              #
        #####################################

        # use BART and maxpooling for encoding
        embedding1 = self.encode(sequences1, self.bart)
        embedding2 = self.encode(sequences2, self.bart)
        embedding3 = self.encode(sequences3, self.bart)
        embedding4 = self.encode(sequences4, self.bart)

        #####################################
        #        Merging Modalities         #
        #####################################

        # individual representations
        out1 = self.fc1(embedding1)
        out2 = self.fc2(embedding2)
        out3 = self.fc3(embedding3)
        out4 = self.fc4(embedding4)

        # first step
        out5 = self.fc5(torch.cat((out1, out2), dim=1))

        # second step
        out6 = self.fc6(torch.cat((out5, out3), dim=1))

        # third step & make the final prediction
        out = self.fc_last(torch.cat((out6, out4), dim=1))

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

    def order_input(self, model, tweet_text, location_description, image_caption, annotation_question):

        acronym_dict = {
            'T': tweet_text,
            'L': location_description,
            'I': image_caption,
            'A': annotation_question
        }

        inputs = model.split("_")[1:]
        input1, input2 = inputs[0].split('and')
        input3, input4 = inputs[1], inputs[2]

        inputs_set = {input1, input2, input3, input4}
        if len(inputs_set) < 4:
            raise Exception("Please choose correct model type")

        try:
            # model takes the sequence1 and sequence2 firstly
            sequence1, sequence2 = acronym_dict[input1], acronym_dict[input2]

            # then sequence3
            sequence3 = acronym_dict[input3]

            # then sequence4
            sequence4 = acronym_dict[input4]
        except Exception:
            raise Exception("Please choose correct model type")

        return sequence1, sequence2, sequence3, sequence4

