from google import genai
from google.genai.types import GenerateContentConfig
from google.genai import types
import os
import numpy as np
from tqdm import tqdm
import argparse
import json
import jsonlines
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from utils.learning_helper.preprocess import Preprocessor

PROJECT_ID = "xxxx" 
LOCATION = "xxx"  
from utils.annotation_filter.annotation_helper import map_prediction


def llava_interface(args, client, model_name):

    qs = args['query']
    print(qs, args['image_file'])
    messages = [
    {"role": "system", "content": "You are a helpful assistant for inferring the duration of human activities."},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": args['image_file'],
            },
            {"type": "text", "text": qs},
        ],
    }
    ]
    with open(args['image_file'], 'rb') as f:
      image_bytes = f.read()
    # image_bytes = requests.get(args['image_file']).content
    # Inference: Generation of the output
    response = client.models.generate_content(
        model = model_name,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/png',
            ),
            qs
        ],
        # config=GenerateContentConfig(
        #     system_instruction=[
        #         "You are a helpful assistant to infer the duration of human activities. Note that most of the tweets were posted during the 2020 COVID-19 pandemic, when extended lockdowns were common and quarantine periods often lasting over 30 days, except for trip purposes."
        #     ]
        # ),
    )

    # input_token_len = input_ids.shape[1]
    # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    # outputs = outputs.strip()
    # if outputs.endswith(stop_str):
    #     outputs = outputs[: -len(stop_str)]
    # outputs = outputs.strip()
    return response.text


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
        model = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        # make predictions
        pbar = tqdm(total=len(test_instances), desc='Evaluating  ')
        for instance in test_instances:
            # prepare the text inputs
            location = instance['location']
            tweet_text = instance['metadata']['full_text']
            # prompt = (
            #     f"The author of the tweet was at {location}. "
            #     f"Determine how long the author will stay at the city after posting that tweet.\n\n"
            #     f"Tweet text:{tweet_text}\n\n"
            #     "OPTIONS:\n"
            #     "1. less than 24 hours.\n"
            #     "2. 1-30 days.\n"
            #     "3. more than 30 days.\n\n"
            #     "Please read the image and give your answer in the following format: Option: Reason\n\n"
            #     "### Example starts ###"
            #     "1. less than 24 hours. The reason is that the image shows a man taking a picture of the bridge and leaving soon. Therefore, the answer is less than 24 hours."
            #     "1. less than 24 hours. The reason is that the Twitter user mentions he watched a soccer yesterday. So he will return home after that. "
            #     "1. less than 24 hours. The reason is that the image shows a man swimming in a pool, about to leave shortly."
            #     "2. 1-30 days. The reason is that the Twitter user is visiting Miami Beach at Independence day, which means he will only stay there for a couple of days. Therefore, the answer is 1-30 days."
            #     "2. 1-30 days. The reason is that the Twitter user is traveling and surfing at Memorial day and will go back after the vacation. Therefore, the answer is 1-30 days."
            #     "3. more than 30 days. The reason is that the image shows a man sitting on the steps of a house, indicating he is living there. Therefore, the answer is more than 30 days."
            #     "3. more than 30 days. The tweet references #quarantinefifteen, it implies that the author might be spending a significant amount of time at home. Given that COVID was ongoing when the tweet was posted, the author may need to spend an extended period at home. "
            #     "### Example ends ###"
            # )
            # multiturn
            prompt = (
                f"The author of the tweet was at {location}. "
                f"Analyze how the tweet's text and hashtags relate to the location.\n\n"
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
            args['image_file'] = instance['image_filepath']
            args['query'] = prompt
            args['conv_mode'] = None
            args['sep'] = ','

            llava_output = llava_interface(args, model, model_name)
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
    with jsonlines.open('annotations/new_annotation.jsonl') as reader:
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
    exclude_label="I cannot determine"
    filtered_labels, filtered_predictions = zip(*[(l, p) for l, p in zip(labels, mapped_predictions) 
                                                  if l != exclude_label and p != exclude_label])
    cm = confusion_matrix(filtered_labels, filtered_predictions, labels=['0-24 hours', '1-30 days', 'more than 30 days'])
    
    # Normalize the confusion matrix by the total number of samples (convert to percentage)
    total_samples = cm.sum()
    cm_percent = cm / total_samples * 100
    print(cm_percent, classification_report(labels, mapped_predictions, digits=3))
    print(classification_report(labels, mapped_predictions, digits=3))
    with open(result_filepath, 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--model', help='')
    parser.add_argument('-output_dir', '--output_dir', help='')
    parser.add_argument('-mode', '--mode', default='evaluate')

    args = parser.parse_args()
    # main(
    #     args.mode,
    #     model_name='gemini-2.5-pro',
    #     output_dir='./experimental_results',
    # )
    main(
        args.mode,
        model_name='gemini-2.5-flash',
        output_dir='./experimental_results',
    )