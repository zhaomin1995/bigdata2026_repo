import os
import re
import torch
import torch.nn as nn
from tqdm import tqdm
from peft import LoraConfig
from transformers import (
    LlamaForCausalLM,
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
)


EXEMPLAR_METADATA = [

    # 1st example
    {
        'label': '3.',
        'location': 'Yucca Valley',
        'tweet_text': "V attended a really engaging class on #covid19 through the #discoversciencecenter. #girlscouts @ Yucca Valley, California",
        'image_caption': 'a woman and a child are using a computer',
        'explanation': "The reason is that the Twitter user is taking a class at Yucca Valley, then most likely he is living there. Therefore, the answer is more than 30 days."
    },

    # 2nd example
    {
        'label': '3.',
        'location': 'Baton Rouge',
        'tweet_text': "Happy Father's Day I love y'all #MYBABIES #HAPPYFATHERSDAY #TOODAREALONES #AMTSHIT🌎 #RIPJALEN👼🏽 #FLYHIGHV👼🏾 #CORONA😷 @ Baton Rouge, Louisiana",
        'image_caption': 'a boy in a red shirt is playing with a toy',
        'explanation': "The reason is that the Twitter user is celebrating Father's day with his family and it seems that they are living together in Baton Rouge. Therefore, the answer is more than 30 days."
    },

    # 3rd example
    {
        'label': '1.',
        'location': 'Brooklyn',
        'tweet_text': "Alone, together. 🚖 🚕 🚖🚕🚖 #nyc #lockdown #lockdown2020 #taxi #alonetogether @ Brooklyn, New York",
        'image_caption': 'a bridge over a body of water',
        'explanation': "The reason is that the Twitter user is taking a picture of the bridge and will leave soon. Therefore, the answer is less than 24 hours."
    },

    # 4th example
    {
        'label': '1.',
        'location': 'Minnetonka',
        'tweet_text': "First patio this summer. And spring. First restaurant this .... forever. #socialdistancing2020 #covid19 #patioseason #wearyourmask😷 earsbyrunshine #familytrip @ Gold Nugget Tavern Grille",
        'image_caption': 'a family of four wearing face masks at a table',
        'explanation': "The reason is that the Twitter user is trying to find a restaurant in the hotel patio. Therefore, the answer is less than 24 hours."
    },

    # 5th example
    {
        'label': '1.',
        'location': 'Clemson',
        'tweet_text': "Botanical garden in Clemson SC. #botanical #garden #spring #travel #quarantine #weather #nature #naturephotography #travelphotography #covid19 #mountains #forest @ South Carolina Botanical Garden",
        'image_caption': 'a woman in a pink shirt is holding an umbrella',
        'explanation': "The reason is that the Twitter user is visiting a botanical garden which takes only a couple hours. Therefore, the answer is less than 24 hours."
    },

    # 6th example
    {
        'label': '1.',
        'location': 'Newmarket',
        'tweet_text': "Outdoor lunch! Yum! #newmarketnh theoakhousenh #localbusiness #lovewhereyouwork #socialdistancing is easy thebigbeancafe_nh",
        'image_caption': 'a series of photos showing a street scene with buildings',
        'explanation': "The reason is that the Twitter user is just exploring different restaurants. He will leave there after enjoying his food. Therefore, the answer is less than 24 hours."
    },

    # 7th example
    {
        'label': '2.',
        'location': 'Miami Beach',
        'tweet_text': "Ready to ride in the weekend! #tgif #friyay #usabirthday #4thofjuly #fireworks #dontmixandmatch #handsanitizer #firecracker #advise #toys #dolls",
        'image_caption': 'a person wearing a helmet with a cup in their mouth',
        'explanation': "The reason is that the Twitter user is visiting Miami Beach at Independence day, which means he will only stay there for a couple of days. Therefore, the answer is 1-30 days."
    },

    # 8th example
    {
        'label': '2.',
        'location': 'Metairie',
        'tweet_text': "Red, White Boo 😍 All photo and caption credits go to @AshleyAmoss #redwhiteandblue #maboo #corona #coronasummer #coronavirus #pandemic #memorialday @ Old Metairie Area",
        'image_caption': 'a man and a woman sitting on a surfboard in the water',
        'explanation': "The reason is that the Twitter user is traveling and surfing at Memorial day and will go back after the vacation. Therefore, the answer is 1-30 days."
    },

    # 9th example
    {
        'label': '3.',
        'location': 'Mckinney',
        'tweet_text': "Ganpati 2020 was a quiet affair... Last year, our house was filled with loved ones, family and friends... we missed them this year. #ganeshfestival2020 #ganpatibappamorya #stayathome #staysafe #fightagainstcorona",
        'image_caption': 'a vase filled with flowers sitting on a table',
        'explanation': "The reason is that the Twitter user mentioned 'our house', indicating he is living there. Therefore, the answer is more than 30 days."
    },

    # 10th example
    {
        'label': '3.',
        'location': 'Chicago',
        'tweet_text': "Last time I saw these guys was the week before everything shut down back in March (damn u, #corona !) I've missed them terribly. Felt great to reunite safely after all these months and have a #socialdistance #picnic",
        'image_caption': 'a large group of people sitting around a picnic table',
        'explanation': "The reason is that the Twitter user mentioned 'Last time I saw these guys', indicating he have been living there for a long time. Therefore, the answer is more than 30 days."
    },

]

EXEMPLARS = [
    (
        f"Read the tweet below. The author of the tweet was at {metadata['location']}. "
        f"Determine how long the author will stay there after the tweet is posted. "
        f"\n\n"
        f"TWEET TEXT: {metadata['tweet_text']}\n"
        f"TWEET IMAGE CAPTION: {metadata['image_caption']}\n\n"
        f"OPTIONS:\n"
        "1. less than 24 hours.\n"
        "2. 1-30 days.\n"
        "3. more than 30 days.\n"
        f"ANSWER: {metadata['label']}"
    )
    for metadata in EXEMPLAR_METADATA
]


EXEMPLARS_COT = [
    (
        f"Read the tweet below. The author of the tweet was at {metadata['location']}. "
        f"Determine how long the author will stay there after the tweet is posted. "
        f"\n\n"
        f"TWEET TEXT: {metadata['tweet_text']}\n"
        f"TWEET IMAGE CAPTION: {metadata['image_caption']}\n\n"
        f"OPTIONS:\n"
        "1. less than 24 hours.\n"
        "2. 1-30 days.\n"
        "3. more than 30 days.\n"
        f"ANSWER: {metadata['label']} {metadata['explanation']}"
    )
    for metadata in EXEMPLAR_METADATA
]


def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#(\S+)', r'\1', text)  # remove the leading # in hashtag
    text = re.sub(r'@(\S+)', r'\1', text)  # remove the leading @ in mention
    return text.strip()


def func_prompt(data_type, prompt, exemplar, label, cot):

    if not cot:
        if exemplar == 'one-shot':
            instruction = '\n\n'.join(EXEMPLARS[:1])
        if exemplar == 'five-shot':
            instruction = '\n\n'.join(EXEMPLARS[:5])
        if exemplar == 'ten-shot':
            instruction = '\n\n'.join(EXEMPLARS[:10])
    else:
        if exemplar == 'one-shot':
            instruction = '\n\n'.join(EXEMPLARS_COT[:1])
        if exemplar == 'five-shot':
            instruction = '\n\n'.join(EXEMPLARS_COT[:5])
        if exemplar == 'ten-shot':
            instruction = '\n\n'.join(EXEMPLARS_COT[:10])

    model_input = ''
    if exemplar != 'zero-shot':
        model_input += f"### Instruction: {instruction}\n"
    if data_type == 'train':
        model_input += f"### Prompt: {prompt}"
        model_input += label
    if data_type == 'test':
        model_input += f"### Prompt: {prompt}"
    return model_input


def get_prompt(instance, model_name, data_type, exemplar, cot=False):
    tweet_text = instance['metadata']['full_text']
    image_caption = instance['metadata']['blip2_caption']
    location = instance['location']
    label = instance['adjudicated_label']
    prompt = (
        f"Read the tweet below. The author of the tweet was at {location}. "
        f"Determine how long the author will stay there after the tweet is posted. "
        f"\n\n"
        f"TWEET TEXT: {tweet_text}\n"
        f"TWEET IMAGE CAPTION: {image_caption}\n\n"
        "OPTIONS:\n"
        "1. less than 24 hours.\n"
        "2. 1-30 days.\n"
        "3. more than 30 days.\n\n"
        "ANSWER: "
    )
    if 'llama' in model_name:
        prompt = 'You are a helpful, respectful and honest assistant. ' + prompt
    return func_prompt(data_type, prompt, exemplar, label, cot)


def inference(model, tokenizer, test_samples, batch_size=1):

    predictions, labels = [], []
    tokenizer.pad_token = tokenizer.eos_token
    pbar = tqdm(total=len(test_samples['text']), desc='Evaluating  ')
    with torch.inference_mode():
        for start in range(0, len(test_samples['text']), batch_size):
            end = min(start + batch_size, len(test_samples['text']))
            texts = [sample for sample in test_samples['text'][start: end]]
            input_ids = tokenizer(
                texts,
                return_tensors='pt',
                max_length=2048,
                padding=True,
                truncation=True,
            ).to(model.device)
            output_tokens = model.generate(**input_ids, max_new_tokens=20, do_sample=False, use_cache=True)
            for ele in output_tokens:
                decoded_output = tokenizer.decode(ele, skip_special_tokens=True)
                decoded_output = decoded_output.replace("<pad>", "").strip()
                predictions.append(decoded_output)
            for label in test_samples['label'][start: end]:
                labels.append(label)
            pbar.update(end - start)
    pbar.close()

    return predictions, labels


def label_mapping(model_name, exemplar, pred):

    if 'llama' not in model_name:
        if pred.startswith('1') or 'less than 24 hours' in pred:
            mapped_label = '0-24 hours'
        elif pred.startswith('2') or '1-30 days' in pred:
            mapped_label = '1-30 days'
        elif pred.startswith('3') or 'more than 30 days' in pred:
            mapped_label = 'more than 30 days'
        else:
            mapped_label = 'I cannot determine'

    else:
        start_indices = [i.start() for i in re.finditer('ANSWER: ', pred)]
        if exemplar == 'zero-shot':
            start_index = start_indices[0]
        elif exemplar == 'one-shot':
            start_index = start_indices[1]
        elif exemplar == 'five-shot':
            start_index = start_indices[5]
        elif exemplar == 'ten-shot':
            start_index = start_indices[10]

        selected_option = pred[start_index + 8: start_index + 9]
        larger_search_scope = pred[start_index + 8: start_index + 38]

        if selected_option == '1' or 'less than 24 hours' in larger_search_scope:
            mapped_label = '0-24 hours'
        elif selected_option == '2' or '1-30 days' in larger_search_scope:
            mapped_label = '1-30 days'
        elif selected_option == '3' or 'more than 30 days' in larger_search_scope:
            mapped_label = 'more than 30 days'
        else:
            mapped_label = 'I cannot determine'

    return mapped_label


def llama_inference(model, tokenizer, test_samples, batch_size=1):

    predictions, labels = [], []
    tokenizer.pad_token = tokenizer.eos_token
    pbar = tqdm(total=len(test_samples['text']), desc='Evaluating  ')
    with torch.inference_mode():
        for start in range(0, len(test_samples['text']), batch_size):
            end = min(start + batch_size, len(test_samples['text']))
            texts = [sample for sample in test_samples['text'][start: end]]
            input_ids = tokenizer(
                texts,
                return_tensors='pt',
                max_length=4096,
                padding=True,
                truncation=True,
            ).to(model.device)
            output_tokens = model.generate(
                **input_ids,
                max_new_tokens=20,
                do_sample=False,
                num_beams=1,
                use_cache=True
            )
            for ele in output_tokens:
                decoded_output = tokenizer.decode(ele, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                predictions.append(decoded_output)
            for label in test_samples['label'][start: end]:
                labels.append(label)
            pbar.update(end - start)
    pbar.close()

    return predictions, labels


def get_model(model_name, cache_dir='/mnt/DATA/hf_cache/'):
    if '+' in model_name:
        model_name = model_name.replace("+", "")
    if model_name == 't5':
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-xxl",
            device_map="auto",
            load_in_8bit=True,
            cache_dir=cache_dir
        )
    elif model_name == 'ul2':
        tokenizer = AutoTokenizer.from_pretrained('google/flan-ul2')
        model = AutoModelForSeq2SeqLM.from_pretrained(
            'google/flan-ul2',
            load_in_8bit=True,  # if we have enough GPU memory, we can set load_in_8bit as False
            device_map="auto",
            cache_dir=cache_dir
        )
    elif model_name == 'alpaca':
        tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-xxl")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "declare-lab/flan-alpaca-xxl",
            load_in_8bit=True,
            device_map="auto",
            cache_dir=cache_dir
        )
    elif model_name == 'llama2chat':
        model = LlamaForCausalLM.from_pretrained(
            'meta-llama/Llama-2-13b-chat-hf',
            load_in_8bit=True,
            device_map='auto',
            cache_dir='/mnt/DATA/hf_cache/'
        )
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b-chat-hf')
    else:
        raise ValueError("Please select the correct model - t5 or ul2 or alpaca")
    return tokenizer, model


def get_train_args(output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        bf16=True,
        learning_rate=2e-5,
        logging_strategy='no',
        # logging_steps=50,
        save_strategy="no",
        # save_steps=50,
        # save_total_limit=5,
        # num_train_epochs=1,
        max_steps=840,
    )
    return training_args


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


def prepare_model_for_training(model):
    """
    This function is for preparing the model loaded in 8bit but training in 32 float precision
    :param model:
    :return:
    """
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = False

    return model


def prepare_model_for_inference(model):
    """
    This function is for preparing the model for inference
    :param model:
    :return:
    """
    for param in model.parameters():
        param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    return model

