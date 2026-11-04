import torch
import torch.nn as nn
from transformers import BartTokenizer, BartModel

# parameter setting
bart_emb_dim = 768
hidden_dim = 128
output_dim = 3


class NBSimplify2TwoInputs(nn.Module):

    def __init__(self, model):
        super(NBSimplify2TwoInputs, self).__init__()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.bart = BartModel.from_pretrained("facebook/bart-base").to(self.device)
        self.fc_last = nn.Linear(3 * bart_emb_dim, output_dim).to(self.device)

    def forward(self, instances):
        #####################################
        #           Get Metadata            #
        #####################################

        sequences1 = []
        sequences2 = []
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
            sequence1, sequence2 = self.order_input(
                self.model, tweet_text, location_description, image_caption, annotation_question
            )
            sequences1.append(sequence1)
            sequences2.append(sequence2)

        #####################################
        #             Encoding              #
        #####################################

        # use BART and maxpooling for encoding
        embedding1 = self.encode(sequences1, self.bart)
        embedding2 = self.encode(sequences2, self.bart)

        #####################################
        #        Merging Modalities         #
        #####################################

        embeddings = [embedding1, embedding2]
        interactions = self.feature_interactions(embeddings)

        # make the final prediction
        out = self.fc_last(torch.cat((embedding1, embedding2, interactions), dim=1))

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

        inputs = model.split("_")[1]
        input1, input2 = list(inputs)
        inputs_set = {input1, input2}
        if len(inputs_set) != 2:
            raise Exception("Please choose correct model type")

        try:
            # model takes the sequence1 and sequence2 firstly
            sequence1, sequence2 = acronym_dict[input1], acronym_dict[input2]

        except Exception:
            raise Exception("Please choose correct model type")

        return sequence1, sequence2

    def feature_interactions(self, embeddings):

        result = []
        embedding_dim, num_embeddings = embeddings[0].shape[1], len(embeddings)
        if embedding_dim >= 2:
            for i in range(num_embeddings - 1):
                for j in range(i + 1, num_embeddings):
                    interaction = embeddings[i] * embeddings[j]
                    result.append(interaction)

        if embedding_dim >= 3:
            for i in range(num_embeddings - 2):
                for j in range(i + 1, num_embeddings - 1):
                    for k in range(j + 1, num_embeddings):
                        interaction = embeddings[i] * embeddings[j] * embeddings[k]
                        result.append(interaction)

        if embedding_dim >= 4:
            for i in range(num_embeddings - 3):
                for j in range(i + 1, num_embeddings - 2):
                    for k in range(j + 1, num_embeddings - 1):
                        for m in range(k + 1, num_embeddings):
                            interaction = embeddings[i] * embeddings[j] * embeddings[k] * embeddings[m]
                            result.append(interaction)

        result = torch.stack(result)
        result = result.permute(1, 0, 2)
        result = torch.flatten(result, start_dim=1)

        return result

