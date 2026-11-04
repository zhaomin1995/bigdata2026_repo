import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, ViTModel


# parameter setting
vit_emb_dim = 768
hidden_dim = 128
output_dim = 3


class ImageOnly(nn.Module):

    def __init__(self, model):
        super(ImageOnly, self).__init__()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.image_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(self.device)
        self.fc1 = nn.Linear(vit_emb_dim, hidden_dim).to(self.device)

    def forward(self, instances):

        #####################################
        #           Get Metadata            #
        #####################################

        tweet_images = [Image.open(instance['image_filepath']).convert('RGB') for instance in instances]

        #####################################
        #             Encoding              #
        #####################################

        # use Visual Transformer
        model_inputs = self.image_processor(tweet_images, return_tensors="pt").to(self.device)
        model_outputs = self.image_model(**model_inputs).pooler_output

        #####################################
        #          Classification           #
        #####################################

        # first step
        out = self.fc1(model_outputs)

        return out
