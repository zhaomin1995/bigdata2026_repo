import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel


torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


# parameter setting
clip_text_emb_dim = 512
clip_image_emb_dim = 768
hidden_dim = 256
output_dim = 3


class CLIPEncoder2(nn.Module):

    def __init__(self, model, dropout_rate=0.2):
        super(CLIPEncoder2, self).__init__()
        self.model = model.split("_")[1]
        if len(self.model) != len(set(self.model)) or not set(self.model).issubset({'T', 'I', 'L'}):
            raise Exception("Please choose correct model type")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        concat_dim = 0
        self.clip_texttokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_textmodel = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_visionmodel = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)

        # we use the BERTweet base as tweet text encoder
        if 'T' in self.model:
            concat_dim += clip_text_emb_dim
            self.w_T = nn.Linear(clip_text_emb_dim, 1).to(self.device)

        # we use Resnet-152 as image encoder
        if 'I' in self.model:
            concat_dim += clip_image_emb_dim
            self.w_I = nn.Linear(clip_image_emb_dim, 1).to(self.device)

        # we use Longformer as location description encoder
        if 'L' in self.model:
            concat_dim += clip_text_emb_dim
            self.w_L = nn.Linear(clip_text_emb_dim, 1).to(self.device)

        self.softmax = nn.Softmax(dim=1)
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
        w = torch.zeros(batch_size, 1).to(self.device)

        # use BERTweet to encode tweet text
        if 'T' in self.model:
            text_inputs = self.clip_texttokenizer(tweet_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            text_outputs = self.clip_textmodel(**text_inputs)
            text_outputs = text_outputs.pooler_output
            w_t = self.w_T(text_outputs)
            w = torch.cat((w, w_t), dim=1)

        # use ResNet to encode the image
        if 'I' in self.model:
            tweet_images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
            tweet_images = self.clip_image_processor(images=tweet_images, return_tensors="pt").to(self.device)
            tweet_images = self.clip_visionmodel(**tweet_images)
            tweet_images = tweet_images.pooler_output
            w_i = self.w_I(tweet_images)
            w = torch.cat((w, w_i), dim=1)

        # use Longformer to encode the location description
        if 'L' in self.model:
            desc_inputs = self.clip_texttokenizer(location_descriptions, padding=True, truncation=True, return_tensors="pt").to(self.device)
            desc_outputs = self.clip_textmodel(**desc_inputs)
            desc_outputs = desc_outputs.pooler_output
            w_l = self.w_L(desc_outputs)
            w = torch.cat((w, w_l), dim=1)

        #####################################
        #        Merging Modalities         #
        #####################################

        # merge modalities and make final prediction
        concat_tensor = concat_tensor[:, 1:]
        w = w[:, 1:]
        w = self.softmax(w)

        # do the multiplication
        model_name = list(self.model)
        while len(model_name) > 0:
            curr_modal = model_name.pop(0)
            if curr_modal == 'T':
                text_outputs = text_outputs * w[:, 0].unsqueeze(1)
                w = w[:, 1:]
                concat_tensor = torch.cat((concat_tensor, text_outputs), dim=1)
            if curr_modal == 'I':
                tweet_images = tweet_images * w[:, 0].unsqueeze(1)
                w = w[:, 1:]
                concat_tensor = torch.cat((concat_tensor, tweet_images), dim=1)
            if curr_modal == 'L':
                desc_outputs = desc_outputs * w[:, 0].unsqueeze(1)
                w = w[:, 1:]
                concat_tensor = torch.cat((concat_tensor, desc_outputs), dim=1)

        out = self.fc1(concat_tensor)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.fc_final(out)

        return out
