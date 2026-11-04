import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, RobertaModel, ViTFeatureExtractor, ViTModel


torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


# parameter setting
roberta_text_emb_dim = 768
vit_image_emb_dim = 768
hidden_dim = 256
output_dim = 3


class IndividualEncoder(nn.Module):

    def __init__(self, model, dropout_rate=0.2):

        super(IndividualEncoder, self).__init__()
        self.model = model.split("_")[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize the preprocessors and pretrained models
        self.text_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.image_processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.text_model = RobertaModel.from_pretrained("roberta-base").to(self.device)
        self.image_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(self.device)

        # get the dimension of the representation of the merged modalities
        if self.model == 'concatenate':
            self.merged_dim = roberta_text_emb_dim + vit_image_emb_dim
        elif self.model == 'addition':
            self.merged_dim = min(roberta_text_emb_dim, vit_image_emb_dim)
        elif self.model == 'multiplication':
            self.merged_dim = min(roberta_text_emb_dim, vit_image_emb_dim)
        elif self.model == 'crossattention':
            self.merged_dim = 2 * min(roberta_text_emb_dim, vit_image_emb_dim)
            self.mha_t_i = nn.MultiheadAttention(min(roberta_text_emb_dim, vit_image_emb_dim), num_heads=1)
            self.mha_i_t = nn.MultiheadAttention(min(roberta_text_emb_dim, vit_image_emb_dim), num_heads=1)
        else:
            raise ValueError("Please choose the correct type of merging: "
                             "concatenate OR addition OR multiplication OR crossattention")

        # initialize the dense layer for classification
        self.fc1 = nn.Linear(self.merged_dim, hidden_dim).to(self.device)
        self.fc_final = nn.Linear(hidden_dim, output_dim).to(self.device)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, instances):

        #####################################
        #           Get Metadata            #
        #####################################

        tweet_texts = []
        image_paths = []
        for instance in instances:

            # get tweet text from the json file
            tweet_text = instance['metadata']['full_text']
            tweet_texts.append(tweet_text)

            # get image paths from the json file
            image_path = instance['image_filepath']
            image_paths.append(image_path)

        #####################################
        #             Encoding              #
        #####################################

        # use RoBERTa to encode tweet text
        text_inputs = self.text_tokenizer(tweet_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        text_outputs = self.text_model(**text_inputs)
        text_outputs = text_outputs.pooler_output

        # use Vision Transformer to encode the image
        tweet_images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        tweet_images = self.image_processor(images=tweet_images, return_tensors="pt").to(self.device)
        tweet_images = self.image_model(**tweet_images)
        tweet_images = tweet_images.pooler_output

        #####################################
        #        Merging Modalities         #
        #####################################

        # merge modalities
        if self.model == 'concatenate':
            merged_representation = torch.cat((text_outputs, tweet_images), dim=1)
        if self.model == 'addition':
            merged_representation = text_outputs + tweet_images
        if self.model == 'multiplication':
            merged_representation = text_outputs * tweet_images
        if self.model == 'crossattention':
            x_t2i, _ = self.mha_t_i(tweet_images, text_outputs, text_outputs)  # cross-attention: text to image
            x_i2t, _ = self.mha_i_t(text_outputs, tweet_images, tweet_images)  # cross-attention: image to text
            merged_representation = torch.cat((x_t2i, x_i2t), dim=1)

        # make the final prediction
        out = self.fc1(merged_representation)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.fc_final(out)

        return out
