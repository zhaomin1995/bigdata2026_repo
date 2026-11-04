import torch
import torch.nn as nn
from transformers import BartTokenizer, BartModel

# parameter setting
bart_emb_dim = 768
output_dim = 3


class NoMatcherModel(nn.Module):

    def __init__(
            self,
            use_loc_desc=True,
            use_image=True,
            use_user_desc=True,
            freeze_bart=False,
            pooling='maxpooling'
    ):
        super(NoMatcherModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.bart_model = BartModel.from_pretrained("facebook/bart-base").to(self.device)
        if freeze_bart:
            for param in self.bart_model.base_model.parameters():
                param.requires_grad = False
        self.use_loc_desc = use_loc_desc
        self.use_image = use_image
        self.use_user_desc = use_user_desc
        self.pooling = pooling
        self.conv1 = nn.Conv1d(256, 1, 1).to(self.device)
        self.fc_tweet1 = nn.Linear(bart_emb_dim, bart_emb_dim).to(self.device)
        self.fc_tweet2 = nn.Linear(bart_emb_dim, output_dim).to(self.device)

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
            location_name = instance['metadata']['place']['full_name']
            annotation_question = f"How long will the author of the tweet " \
                                  f"continuously stay at {location_name} after publishing this tweet?"

            # concatenate all text
            inputs = [annotation_question, tweet_text]
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

        # use BART to encode the concatenated sequence
        bart_inputs = self.bart_tokenizer(matcher_input, return_tensors="pt", truncation=True, padding='max_length',
                                          max_length=256).to(self.device)
        bart_outputs = self.bart_model(**bart_inputs)
        bart_last_hidden_states = bart_outputs.last_hidden_state

        # max-pooling to get the representation of each sequence
        if self.pooling == 'maxpooling':
            sequence_length = bart_last_hidden_states.shape[1]
            bart_pooled_output = nn.MaxPool1d(kernel_size=sequence_length)(bart_last_hidden_states.permute(0, 2, 1))
            bart_pooled_output = bart_pooled_output.squeeze(2)
        elif self.pooling == 'meanpooling':
            sequence_length = bart_last_hidden_states.shape[1]
            bart_pooled_output = nn.AvgPool1d(kernel_size=sequence_length)(bart_last_hidden_states.permute(0, 2, 1))
            bart_pooled_output = bart_pooled_output.squeeze(2)
        elif self.pooling == 'convolution':
            bart_pooled_output = self.conv1(bart_last_hidden_states)
            bart_pooled_output = bart_pooled_output.squeeze(1)
        else:
            raise Exception("Please choose the correct pooling strategy: 'maxpooling', 'meanpooling', or 'convolution'")

        # add fully-connected layer after max-pooling
        output = self.fc_tweet1(bart_pooled_output)
        output = self.fc_tweet2(output)

        return output
