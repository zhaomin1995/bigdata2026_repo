import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from transformers import LongformerTokenizer, LongformerModel, ViTFeatureExtractor, ViTModel


torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


# parameter setting
bertweet_emb_dim = 768
vit_emb_dim = 768
longformer_emb_dim = 768
hidden_dim = 256
output_dim = 3


class MultiEncoder3(nn.Module):

    def __init__(self, model, dropout_rate=0.2):
        super(MultiEncoder3, self).__init__()
        self.model = model.split("_")[1]
        if len(self.model) != len(set(self.model)) or not set(self.model).issubset({'T', 'I', 'L'}):
            raise Exception("Please choose correct model type")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        concat_dim = 0

        # we use the BERTweet base as tweet text encoder
        if 'T' in self.model:
            self.bertweet_tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
            self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base").to(self.device)
            self.bertweet_max_length = 128
            concat_dim += bertweet_emb_dim

        # we use Resnet-152 as image encoder
        if 'I' in self.model:
            self.image_preprocess = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
            self.vitmodel = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(self.device)
            concat_dim += vit_emb_dim

        if 'L' in self.model:
            # we use Longformer as location description encoder
            self.longformer = LongformerModel.from_pretrained("allenai/longformer-base-4096").to(self.device)
            self.longformer_tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
            self.longformer_max_length = 512
            concat_dim += longformer_emb_dim

        self.fc1 = nn.Linear(concat_dim, hidden_dim).to(self.device)
        self.fc_final = nn.Linear(hidden_dim, output_dim).to(self.device)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, instances):

        #####################################
        #           Get Metadata            #
        #####################################

        tweet_texts = []
        location_descriptions = []
        image_paths = []
        for instance in instances:
            # get everything from the json file
            if 'T' in self.model:
                tweet_text = instance['metadata']['full_text']
                tweet_texts.append(tweet_text)
            if 'L' in self.model:
                location_description = instance['metadata']['location_description']
                location_descriptions.append(location_description)
            if 'I' in self.model:
                image_path = instance['image_filepath']
                image_paths.append(image_path)

        #####################################
        #             Encoding              #
        #####################################

        batch_size = len(instances)
        concat_tensor = torch.zeros(batch_size, 1).to(self.device)

        # use BERTweet to encode tweet text
        if 'T' in self.model:
            input_ids = []
            attention_mask = []
            for tweet_text in tweet_texts:
                encoding = self.bertweet_tokenizer(tweet_text, max_length=self.bertweet_max_length, padding='max_length')
                input_ids.append(encoding['input_ids'])
                attention_mask.append(encoding['attention_mask'])
            input_ids = torch.tensor(input_ids).to(self.device)
            attention_mask = torch.tensor(attention_mask).to(self.device)
            bertweet_output = self.bertweet(input_ids=input_ids, attention_mask=attention_mask)
            bertweet_output = bertweet_output.pooler_output
            concat_tensor = torch.cat((concat_tensor, bertweet_output), dim=1)

        # use ResNet to encode the image
        if 'I' in self.model:
            tweet_images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
            tweet_images = self.image_preprocess(tweet_images, return_tensors="pt").to(self.device)
            tweet_images = self.vitmodel(**tweet_images)
            tweet_images = tweet_images.pooler_output
            concat_tensor = torch.cat((concat_tensor, tweet_images), dim=1)

        # use Longformer to encode the location description
        if 'L' in self.model:
            input_ids = []
            attention_mask = []
            for location_description in location_descriptions:
                encoding = self.longformer_tokenizer(location_description, max_length=self.longformer_max_length,
                                                     padding='max_length')
                input_ids.append(encoding['input_ids'])
                attention_mask.append(encoding['attention_mask'])
            input_ids = torch.tensor(input_ids).to(self.device)
            attention_mask = torch.tensor(attention_mask).to(self.device)
            longformer_output = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
            longformer_output = longformer_output.pooler_output
            concat_tensor = torch.cat((concat_tensor, longformer_output), dim=1)

        #####################################
        #        Merging Modalities         #
        #####################################

        # merge modalities and make final prediction
        concat_tensor = concat_tensor[:, 1:]
        out = self.fc1(concat_tensor)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.fc_final(out)

        return out
