import openai


import os
os.environ["HUGGINGFACE_HUB_CACHE"] = '/mnt/DATA/hf_cache/'

import argparse
import json
import jsonlines
from tqdm import tqdm
from pathlib import Path

from sklearn.metrics import classification_report
from utils.learning_helper.preprocess import Preprocessor
client = openai.OpenAI(api_key="xxxxx", base_url="https://openkey.cloud/v1")
from utils.annotation_filter.annotation_helper import map_prediction


def llava_interface(args, model_name):

    qs = args['query']
    messages = [
    {"role": "system", "content": "You are a helpful assistant for inferring the duration of human activities."},
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": args['image_file']
                },
            },
            {"type": "text", "text": qs},
        ],
    }
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    outputs = response.choices

    # input_token_len = input_ids.shape[1]
    # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    # outputs = outputs.strip()
    # if outputs.endswith(stop_str):
    #     outputs = outputs[: -len(stop_str)]
    # outputs = outputs.strip()
    return outputs[0].message.content


def main(
        mode,
        model_name,
        output_dir,
        result_foldername='results',
        response_foldername='responses',
):
    response_folder = os.path.join(output_dir, response_foldername)
    Path(response_folder).mkdir(parents=True, exist_ok=True)
    response_filename = f"{model_name}_response"
    response_filepath = os.path.join(response_folder, response_filename)
    responses, mapped_predictions = [], []
    # initialize the helpers
    preprocessor = Preprocessor()
    if mode == 'evaluate':

        # load annotations and tweet metadata
        # instances = preprocessor.load_annotations(annotation_folder, tweet_folder)
        instances = preprocessor.load_improved_data('improved_data.json')

        # split instances into train, development, and test
        # we only use test instances if we only do inference
        train_instances, dev_instances, test_instances = preprocessor.split_instances(instances)

        # merge the training data and development data
        train_instances = train_instances + dev_instances
        # make predictions
        pbar = tqdm(total=len(test_instances), desc='Evaluating  ')
        for instance in test_instances:
            # prepare the text inputs
            location = instance['location']
            tweet_text = instance['metadata']['full_text']
            prompt = (
                f"The author of the tweet was at {location}. "
                f"Determine how long the author will stay at the city after posting that tweet.\n\n"
                f"Tweet text:{tweet_text}\n\n"
                "OPTIONS:\n"
                "1. less than 24 hours.\n"
                "2. 1-30 days.\n"
                "3. more than 30 days.\n\n"
                "Please read the image and give your answer in the following format: Option: Reason\n\n"
                "### Example starts ###"
                "1. less than 24 hours. The reason is that the image shows a man taking a picture of the bridge and leaving soon. Therefore, the answer is less than 24 hours."
                "1. less than 24 hours. The reason is that the Twitter user mentions he watched a soccer yesterday. So he will return home after that. "
                "1. less than 24 hours. The reason is that the image shows a man swimming in a pool, about to leave shortly."
                "2. 1-30 days. The reason is that the Twitter user is visiting Miami Beach at Independence day, which means he will only stay there for a couple of days. Therefore, the answer is 1-30 days."
                "2. 1-30 days. The reason is that the Twitter user is traveling and surfing at Memorial day and will go back after the vacation. Therefore, the answer is 1-30 days."
                "3. more than 30 days. The reason is that the image shows a man sitting on the steps of a house, indicating he is living there. Therefore, the answer is more than 30 days."
                "3. more than 30 days. The tweet references #quarantinefifteen, it implies that the author might be spending a significant amount of time at home. Given that COVID was ongoing when the tweet was posted, the author may need to spend an extended period at home. "
                "### Example ends ###"
            )

            # prepare the argument that LLaVa needs
            args = {}
            args['image_file'] = instance['display_link']
            args['query'] = prompt
            args['conv_mode'] = None
            args['sep'] = ','

            llava_output = llava_interface(args, model_name)
            print(llava_output)
            # get the mapped label
            responses.append({"response":llava_output,
                            "id":instance['instance_id']
                            })
            mapped_output = map_prediction(llava_output)
            mapped_predictions.append(mapped_output)
            pbar.update(1)
        pbar.close()
        with open(response_filepath, 'w') as file:
            json.dump(responses, file)
    else:
        with open(response_filepath, 'r') as file:
            responses = json.load(file)
        for item in responses:
            mapped_predictions.append(map_prediction(item['response']))

    # get the list of ground truth
    with jsonlines.open('annotations2/new_annotation.jsonl') as reader:
        instances = [obj for obj in reader]
        # split instances into train, development, and test
        # we only use test instances if we only do inference
        train_instances, dev_instances, test_instances = preprocessor.split_instances(instances)
        labels = [instance['adjudicated_label'] for instance in test_instances]
    # save the evaluation results
    result_folder = os.path.join(output_dir, result_foldername)
    Path(result_folder).mkdir(parents=True, exist_ok=True)
    result_filename = f"{model_name}_result"
    result_filepath = os.path.join(result_folder, result_filename)
    results = classification_report(labels, mapped_predictions, output_dict=True)
    print(classification_report(labels, mapped_predictions, digits=3))
    with open(result_filepath, 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--model', help='')
    parser.add_argument('-output_dir', '--output_dir', help='')
    parser.add_argument('-mode', '--mode', default='evaluate')
    args = parser.parse_args()
    main(
        args.mode,
        model_name='gpt-5-mini', # grok-4
        output_dir='./experimental_results',
    )