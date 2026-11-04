import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel, AutoProcessor, BlipModel


torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


# parameter setting
clip_text_emb_dim = 512
clip_image_emb_dim = 768
blip_text_emb_dim = 512
blip_image_emb_dim = 512
hidden_dim = 256
output_dim = 3


class SharedEncoder(nn.Module):

    def __init__(self, model, dropout_rate=0.2):

        super(SharedEncoder, self).__init__()
        self.model = model.split("_")[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize the preprocessors and pretrained models
        if self.model == 'clip':
            self.clip_texttokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_textmodel = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_visionmodel = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.merged_dim = clip_text_emb_dim + clip_image_emb_dim
        elif self.model == 'blip':
            self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
            self.merged_dim = blip_text_emb_dim + blip_image_emb_dim
        else:
            raise ValueError("Please choose the correct name of the encoder: CLIP or BLIP")

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

        # use shared encoders to encode both tweet text and images
        if self.model == 'clip':
            text_inputs = self.clip_texttokenizer(
                tweet_texts, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            text_outputs = self.clip_textmodel(**text_inputs)
            text_outputs = text_outputs.pooler_output

            tweet_images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
            tweet_images = self.clip_image_processor(images=tweet_images, return_tensors="pt").to(self.device)
            tweet_images = self.clip_visionmodel(**tweet_images)
            tweet_images = tweet_images.pooler_output
        elif self.model == 'blip':
            tweet_images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
            model_inputs = self.blip_processor(
                text=tweet_texts, images=tweet_images, return_tensors="pt", padding=True
            ).to(self.device)
            model_outputs = self.blip_model(**model_inputs)
            text_outputs = model_outputs.text_embeds
            tweet_images = model_outputs.image_embeds
        else:
            raise ValueError("Please choose the correct name of the encoder: CLIP or BLIP")

        #####################################
        #        Merging Modalities         #
        #####################################

        # merge modalities
        merged_representation = torch.cat((text_outputs, tweet_images), dim=1)

        # make the final prediction
        out = self.fc1(merged_representation)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.fc_final(out)

        return out
