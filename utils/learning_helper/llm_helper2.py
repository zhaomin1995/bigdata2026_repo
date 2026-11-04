import os
import re
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoProcessor,
    AutoModelForVision2Seq,
    BatchFeature
)
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms


EXEMPLAR_METADATA = [
    {'location': 'Brooklyn', 'label': '1.', 'tweet_text': "Heather's 1st time back to church since March. You Can't tell behind the mask but she's excited and overwhelmed to finally be back. #Churchflow  #Sundaygathering #socialdistancing #wearyourmask", 'image_filepath': 'data/1294978479423922177/1294978479423922177.png'},
    {'location': 'San Francisco', 'label': '2.', 'tweet_text': '#sanfrancisco #sirfrancisdrake #flowers #coronavirus The art everywhere is refreshing. @ Kimpton Sir Francis Drake Hotel', 'image_filepath': 'data/1261536479676792832/1261536479676792832.png'},
    {'location': 'Queens', 'label': '1.', 'tweet_text': "Rocking on⚡ #livemusic #queensrocks #rock #singlecut #laterraza #queens #covid #outdoors", 'image_filepath': 'data/1303405395705311232/1303405395705311232.png'},
    {'location': 'Atlanta', 'label': '3.', 'tweet_text': "Social distancing at the park like a champ! #aquarantinedaphne #dachshundsofinstagram #doxie #modoxie #dogsofinstagram #socialdistancing @ Piedmont Park", 'image_filepath': 'data/1274409004345720832/1274409004345720832.png'},
    {'location': 'Atlanta', 'label': '3.', 'tweet_text': "We all are starting to miss normal interaction aren’t we ?!  #handsaremeanttobeheld #dove #covid19 #cooper @ Kirkwood", 'image_filepath': 'data/1290626077690408960/1290626077690408960.png'},
    {'location': 'Manhattan', 'label': '2.', 'tweet_text': "The New New York stay safe everyone #StayHumble #StayHungry #MMunroePhotography #covid_19 #coronavirus #quarantinephotos #photography #photo #urban", 'image_filepath': 'data/1262536167553216516/1262536167553216516.png'},
    {'location': 'Washington DC', 'label': '1.', 'tweet_text': "Henry doing his best Jenny re-enactment from Forest Gump! #coronavirus #toddlerlife #lincolnmemorial #memorialday @ Lincoln Memorial", 'image_filepath': 'data/1264663021206425600/1264663021206425600.png'},
    {'location': 'Manhattan', 'label': '1.', 'tweet_text': "Thanks emirates - always keeping it classy and safe! #emirates #uae #dubai #pakistan #flying #hygiene #covid #safetyfirst @ New York, New York", 'image_filepath': 'data/1298449930605010946/1298449930605010946.png'},
    {'location': 'Pacifica', 'label': '1.', 'tweet_text': "A hike through the American Wasteland... #covid #virus #nature #shelterinplace #hike", 'image_filepath': 'data/1247610560142811138/1247610560142811138.png'},
    {'location': 'Leavenworth', 'label': '3.', 'tweet_text': "Put me in coach, I’m ready! #fall #2020 #usm #spires #covid19 #mitigation @ University of Saint Mary", 'image_filepath': 'data/1292917419879436289/1292917419879436289.png'},
    {'location': 'Colesville', 'label': '2.', 'tweet_text': "Dinner outside with a 🔥 Lola took over the comfy double lounge 🤣 #corona #thisis50 #firstnightoutside", 'image_filepath': 'data/1246947263139721217/1246947263139721217.png'},
]


EXEMPLAR_TEXTS = [
    (
        f"Read the tweet below. The author of the tweet was at {metadata['location']}. "
        f"Determine how long the author will stay there after the tweet was posted. "
        f"\n\n"
        f"{metadata['tweet_text']}\n"
        f"<image>\n\n"
        "OPTIONS:\n"
        "1. less than 24 hours.\n"
        "2. 1-30 days.\n"
        "3. more than 30 days.\n\n"
        f"Answer: {metadata['label']}"
    ) for metadata in EXEMPLAR_METADATA
]


def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#(\S+)', r'\1', text)  # remove the leading # in hashtag
    text = re.sub(r'@(\S+)', r'\1', text)  # remove the leading @ in mention
    return text.strip()


def load_images(instances):
    images = []
    for instance in instances:
        images.append(Image.open(instance['image_filepath']).convert('RGB'))
    return images


def process_image(image_processor, image_filepaths):
    vision_x = []
    for image_filepath in image_filepaths:
        instance_image = Image.open(image_filepath).convert('RGB')
        vision_x.append(image_processor(instance_image).unsqueeze(0))
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0).to('cuda')

    return vision_x


def get_flamingo_images(instance, image_processor, exemplar):

    all_image_filepaths = [metadata['image_filepath'] for metadata in EXEMPLAR_METADATA]
    if exemplar == 'one-shot':
        all_image_filepaths = all_image_filepaths[:1]
    elif exemplar == 'five-shot':
        all_image_filepaths = all_image_filepaths[:5]
    elif exemplar == 'ten-shot':
        all_image_filepaths = all_image_filepaths[:10]

    all_image_filepaths = all_image_filepaths + [instance['image_filepath']]  # add the current instance image
    flamingo_images = process_image(image_processor, all_image_filepaths)

    return flamingo_images


def func_prompt(data_type, prompt, exemplar, label):

    model_input = ''
    # if exemplar != 'zero-shot':
    #     model_input += f"### Instruction: {instruction}\n"
    if data_type == 'train':
        model_input += f"### Prompt: {prompt}"
        model_input += label
    if data_type == 'test':
        model_input += f"### Prompt: {prompt}"
    return model_input


def get_prompt(instance, data_type, exemplar):
    tweet_text = clean_text(instance['metadata']['full_text'])
    location = instance['location']
    label = instance['adjudicated_label']
    prompt = (
        f"Read the tweet below and determine how long the author of the tweet "
        f"will stay at {location} after the tweet is posted. "
        f"Be more confident in selecting 3 if the tweet is describing author's daily life. "
        f"\n\n"
        f"TWEET TEXT: {tweet_text}\n\n"
        "OPTIONS:\n"
        "1. less than 24 hours.\n"
        "2. 1-30 days.\n"
        "3. more than 30 days.\n\n"
        "Answer: "
    )
    return func_prompt(data_type, prompt, exemplar, label)


def get_flamingo_inputs(instance, image_processor, tokenizer, exemplar):

    flamingo_images = get_flamingo_images(instance, image_processor, exemplar)

    if exemplar == 'one-shot':
        instruction_texts = '<|endofchunk|>'.join(EXEMPLAR_TEXTS[:1]) + '<|endofchunk|>'
    elif exemplar == 'five-shot':
        instruction_texts = '<|endofchunk|>'.join(EXEMPLAR_TEXTS[:5]) + '<|endofchunk|>'
    elif exemplar == 'ten-shot':
        instruction_texts = '<|endofchunk|>'.join(EXEMPLAR_TEXTS) + '<|endofchunk|>'

    # prepare the text inputs
    location = instance['location']
    tweet_text = clean_text(instance['metadata']['full_text'])
    prompt = (
        f"Read the tweet. The author of the tweet was at {location}. "
        f"Determine how long the author will stay there after posting that tweet.\n\n"
        f"{tweet_text}\n"
        f"<image>\n\n"
        "OPTIONS:\n"
        "1. less than 24 hours.\n"
        "2. 1-30 days.\n"
        "3. more than 30 days.\n\n"
        "ANSWER: "
    )

    # put the instruction before the instance
    if exemplar != 'zero-shot':
        prompt = instruction_texts + prompt

    # get the text embedding
    lang_x = tokenizer([prompt], return_tensors="pt").to('cuda')

    return flamingo_images, lang_x


def get_model(model_name, cache_dir='/mnt/DATA/hf_cache/'):
    if '+' in model_name:
        model_name = model_name.replace("+", "")
    if model_name == 'blip2':
        # load the processor and image
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xxl",
            load_in_8bit=True,
            device_map='auto',
            cache_dir=cache_dir
        )
        return processor, model
    elif model_name == 'flamingo':
        # load the raw model, processor, and tokenizer
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-7b",
            tokenizer_path="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=1,
        )
        # load the pretrained weights and set up the tokenizer
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        model = model.to('cuda')
        tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
        return model, image_processor, tokenizer
    elif model_name == 'kosmos':
        model = AutoModelForVision2Seq.from_pretrained(
            "microsoft/kosmos-2-patch14-224",
            device_map='auto',
            cache_dir='/mnt/DATA/hf_cache/'
        )
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        return processor, model
    else:
        raise ValueError("Please select the correct model - blip2 or flamingo or kosmos")


def process_kosmos_interleaved_data(
        processor,
        prompt,
        images,
        placeholder="<i>",
        num_image_tokens=64,
        add_special_tokens=True,
        add_eos_token=False,
        return_tensors=None
):
    first_image_token_id = processor.tokenizer.unk_token_id + 1

    image_input_ids = [processor.tokenizer.convert_tokens_to_ids(processor.boi_token)] \
                      + list(range(first_image_token_id, num_image_tokens + first_image_token_id)) \
                      + [processor.tokenizer.convert_tokens_to_ids(processor.eoi_token)]
    image_attention_mask = [1] * len(image_input_ids)
    image_embeds_position_mask = [0] + [1] * (len(image_input_ids) - 2) + [0]

    components = re.split(rf"({placeholder})", prompt)

    outputs = {"input_ids": [], "attention_mask": [], "image_embeds_position_mask": []}
    for component in components:
        if component != placeholder:
            # add text tokens: no special tokens -> add them at the end
            encoded = processor(text=component, add_special_tokens=False)
            for key in ["input_ids", "attention_mask"]:
                outputs[key].extend(encoded[key])
            outputs["image_embeds_position_mask"].extend([0] * len(encoded["input_ids"]))
        else:
            # add tokens to indicate image placeholder
            outputs["input_ids"].extend(image_input_ids)
            outputs["attention_mask"].extend(image_attention_mask)
            outputs["image_embeds_position_mask"].extend(image_embeds_position_mask)

    if add_special_tokens:
        outputs["input_ids"] = [processor.tokenizer.bos_token_id] + outputs["input_ids"] + (
            [processor.tokenizer.eos_token_id] if add_eos_token else [])
        outputs["attention_mask"] = [1] + outputs["attention_mask"] + ([1] if add_eos_token else [])
        outputs["image_embeds_position_mask"] = [0] + outputs["image_embeds_position_mask"] + (
            [0] if add_eos_token else [])

    outputs["pixel_values"] = processor.image_processor(images).pixel_values

    for key in ["input_ids", "attention_mask", "image_embeds_position_mask"]:
        outputs[key] = [outputs[key]]
    outputs = BatchFeature(data=outputs, tensor_type=return_tensors)

    return outputs


def map_prediction(model_output):
    # Re-export the centralized mapper from the annotation helper to keep one canonical implementation
    from utils.annotation_filter.annotation_helper import map_prediction
    return map_prediction(model_output)


def get_peft_config(peft_name='lora'):
    if peft_name == 'lora':
        peft_config = LoraConfig(lora_alpha=4, r=2, task_type="CAUSAL_LM")
    elif peft_name == 'prefix-tuning':
        pass  # this is on the research agenda
    elif peft_name == 'prompt-tuning':
        pass  # this is on the research agenda
    else:
        raise ValueError("Please check the name of peft method")
    return peft_config
