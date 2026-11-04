import torch
import torch.nn as nn
from transformers import BartTokenizer, BartModel

# parameter setting
bart_emb_dim = 768
output_dim = 3


class MatcherModel1(nn.Module):

    def __init__(
            self,
            use_loc_desc=True,
            use_image=True,
            use_user_desc=True,
            freeze_bart=False,
            pooling='maxpooling'
    ):
        super(MatcherModel1, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.sequence_encoder = BartModel.from_pretrained("facebook/bart-base").to(self.device)
        if freeze_bart:
            for param in self.sequence_encoder.base_model.parameters():
                param.requires_grad = False
        self.use_loc_desc = use_loc_desc
        self.use_image = use_image
        self.use_user_desc = use_user_desc
        self.pooling = pooling
        self.conv1 = nn.Conv1d(256, 1, 1).to(self.device)
        self.conv2 = nn.Conv2d(256, 1, 1).to(self.device)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # encode the three statements so that the model can directly use it without future encoding
        self.statement_encoder = BartModel.from_pretrained("facebook/bart-base").to(self.device)
        for param in self.statement_encoder.base_model.parameters():
            param.requires_grad = False

    def forward(self, instances):

        #####################################
        #        Merger Implementation      #
        #####################################

        matcher_input = []
        for instance in instances:

            # get the Merger input
            tweet_text = instance['metadata']['full_text']
            location_description = instance['metadata']['location_description']
            user_description = instance['metadata']['user']['description']
            image_caption = instance['metadata']['image_caption']

            # concatenate all text
            inputs = [tweet_text]
            if self.use_user_desc:
                inputs.append(user_description)
            if self.use_image:
                inputs.append(image_caption)
            if self.use_loc_desc:
                inputs.append(location_description)
            concatenated_text = '<SEP>'.join(inputs)

            matcher_input.append(concatenated_text)

        #####################################
        #       Matcher Implementation      #
        #####################################

        # use sentence encoder (frozen BART) to encode three statements
        statement_embeddings = []
        for instance in instances:
            location_name = instance['metadata']['place']['full_name']
            statement_hours = f"The author of the tweet will continuously stay at {location_name} " \
                              f"for less than one day after publishing this tweet."
            statement_days = f"The author of the tweet will continuously stay at {location_name} " \
                             f"for less than one month after publishing this tweet."
            statement_more = f"The author of the tweet will continuously stay at {location_name} " \
                             f"for more than one month after publishing this tweet."
            statements = [statement_hours, statement_days, statement_more]
            bart_inputs_statements = self.bart_tokenizer(statements, return_tensors="pt", truncation=True,
                                                         padding='max_length', max_length=256).to(self.device)
            bart_outputs_statements = self.statement_encoder(**bart_inputs_statements)
            statement_embedding = bart_outputs_statements.last_hidden_state
            statement_embeddings.append(statement_embedding)
        statement_embeddings = torch.stack(statement_embeddings)  # statement_embeddings (batch_size, 3, seq_length, bart_emb)

        # use BART to encode the concatenated sequence
        bart_inputs = self.bart_tokenizer(matcher_input, return_tensors="pt", truncation=True, padding='max_length',
                                          max_length=256).to(self.device)
        bart_outputs = self.sequence_encoder(**bart_inputs)
        bart_last_hidden_states = bart_outputs.last_hidden_state

        # reduce the dimensionality of the embeddings of sequence and statements
        statement_length = bart_last_hidden_states.shape[1]
        if self.pooling == 'maxpooling':

            # max-pool the sequence embeddings
            sequence_length = bart_last_hidden_states.shape[1]
            bart_pooled_output = nn.MaxPool1d(kernel_size=sequence_length)(bart_last_hidden_states.permute(0, 2, 1))
            sequence_pooled_output = bart_pooled_output.squeeze(2)

            # max-pool the statement embeddings
            statement_pooled_output = nn.MaxPool2d(kernel_size=(statement_length, 1))(statement_embeddings)
            statement_pooled_output = statement_pooled_output.squeeze(2)

        elif self.pooling == 'meanpooling':

            # mean-pool the sequence embedding
            sequence_length = bart_last_hidden_states.shape[1]
            bart_pooled_output = nn.AvgPool1d(kernel_size=sequence_length)(bart_last_hidden_states.permute(0, 2, 1))
            sequence_pooled_output = bart_pooled_output.squeeze(2)

            # mean-pool the statement embeddings
            statement_pooled_output = nn.MaxPool2d(kernel_size=(statement_length, 1))(statement_embeddings)
            statement_pooled_output = statement_pooled_output.squeeze(2)

        elif self.pooling == 'convolution':

            # use convolution to reduce the dimensionality of sequence embedding
            bart_pooled_output = self.conv1(bart_last_hidden_states)
            sequence_pooled_output = bart_pooled_output.squeeze(1)

            # use convolution to reduce the dimensionality of statement embedding
            statement_pooled_output = self.conv2(statement_embeddings.permute(0, 2, 1, 3))
            statement_pooled_output = statement_pooled_output.squeeze(1)

        else:
            raise Exception("Please choose the correct pooling strategy: 'maxpooling', 'meanpooling', or 'convolution'")

        # get the embedding of each statement
        hour_statement_embedding = torch.select(statement_pooled_output, dim=1, index=0)
        day_statement_embedding = torch.select(statement_pooled_output, dim=1, index=1)
        more_statement_embedding = torch.select(statement_pooled_output, dim=1, index=2)

        # match the encoded statement and the sequence
        hour_statement_score = self.cos(hour_statement_embedding, sequence_pooled_output)
        day_statement_score = self.cos(day_statement_embedding, sequence_pooled_output)
        more_statement_score = self.cos(more_statement_embedding, sequence_pooled_output)
        scores = torch.stack([hour_statement_score, day_statement_score, more_statement_score])
        scores = scores.permute(1, 0)

        return scores
