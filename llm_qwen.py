import os
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/DATA/hf_cache/"

import argparse
import json
import torch
import jsonlines
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import classification_report
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import re
from utils.learning_helper.preprocess import Preprocessor
from utils.annotation_filter.annotation_helper import map_prediction

def clean_thinking_tokens(raw_text: str) -> str:
    if "</think>" in raw_text:
        # Take everything after the last </think> (in case of nested/multiple)
        return raw_text.split("</think>")[-1].strip()
    return raw_text.strip()
def qwen_interface(args, tokenizer, model, processor):
    if "messages" in args:
        messages = args["messages"]
    else:
        qs = args["query"]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": args["image_file"],
                    },
                    {
                        "type": "text",
                        "text": qs,
                    },
                ],
            }
        ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2100,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    outputs = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return clean_thinking_tokens(outputs[0])


def main(
    mode,
    model_name,
    output_dir,
    result_foldername="results",
    response_foldername="responses",
    multiturn=False,
):
    response_folder = os.path.join(output_dir, response_foldername)
    Path(response_folder).mkdir(parents=True, exist_ok=True)

    response_filename = f"{model_name}_{'multiturn_' if multiturn else ''}response"
    response_filepath = os.path.join(response_folder, response_filename)

    responses = []
    mapped_predictions = []

    preprocessor = Preprocessor()

    if mode == "evaluate":
        instances = preprocessor.load_improved_data("improved_data.json")

        train_instances, dev_instances, test_instances = preprocessor.split_instances(
            instances
        )

        train_instances = train_instances + dev_instances

        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        pbar = tqdm(total=len(test_instances), desc="Evaluating  ")

        for instance in test_instances:
            location = instance["location"]
            tweet_text = instance["metadata"]["full_text"]

            prompt = (
                f"The author of the tweet was at {location}. "
                f"Determine how long the author will stay at the city after posting that tweet.\n\n"
                f"Tweet text:{tweet_text}\n\n"
                "OPTIONS:\n"
                "1. less than 24 hours.\n"
                "2. 1-30 days.\n"
                "3. more than 30 days.\n\n"
                "Please read the typical duration examples and give your answer given the provided image "
                "in the following format: Option: Reason\n\n"
                "### Example starts ###\n"
                "1. less than 24 hours. The reason is that the image shows a man taking a picture "
                "of the bridge and leaving soon. Therefore, the answer is less than 24 hours.\n"
                "1. less than 24 hours. The reason is that the Twitter user mentions he watched "
                "a soccer yesterday. So he will return home after that. \n"
                "1. less than 24 hours. The reason is that the image shows a man swimming in a pool, "
                "about to leave shortly.\n"
                "2. 1-30 days. The reason is that the Twitter user is visiting Miami Beach at "
                "Independence day, which means he will only stay there for a couple of days. "
                "Therefore, the answer is 1-30 days.\n"
                "2. 1-30 days. The reason is that the Twitter user is traveling and surfing at "
                "Memorial day and will go back after the vacation. Therefore, the answer is 1-30 days.\n"
                "3. more than 30 days. The reason is that the image shows a man sitting on the steps "
                "of a house, indicating he is living there. Therefore, the answer is more than 30 days.\n"
                "3. more than 30 days. The tweet references #quarantinefifteen, it implies that the "
                "author might be spending a significant amount of time at home. Given that COVID was "
                "ongoing when the tweet was posted, the author may need to spend an extended period at home. \n"
                "### Example ends ###"
            )

            if multiturn:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant for inferring the duration of human activities."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": instance["display_link"]},
                            {"type": "text", "text": f"Describe the tweet image in detail, and analyze its relationship to the tweet text and hashtags.\n\nTweet text: {tweet_text}"}
                        ]
                    }
                ]
                qwen_output_turn1 = qwen_interface({"messages": messages}, tokenizer, model, processor)
                messages.append({"role": "assistant", "content": qwen_output_turn1})
                messages.append({"role": "user", "content": prompt})
                qwen_output = qwen_interface({"messages": messages}, tokenizer, model, processor)
            else:
                args = {
                    "image_file": instance["display_link"],
                    "query": prompt,
                    "conv_mode": None,
                    "sep": ",",
                }
                qwen_output = qwen_interface(args, tokenizer, model, processor)
            # print(qwen_output)

            responses.append(
                {
                    "response": qwen_output,
                    "id": instance["instance_id"],
                }
            )

            mapped_output = map_prediction(qwen_output)
            mapped_predictions.append(mapped_output)

            pbar.update(1)

        pbar.close()

        with open(response_filepath, "w") as file:
            json.dump(responses, file)

    else:
        with open(response_filepath, "r") as file:
            responses = json.load(file)

        for item in responses:
            mapped_predictions.append(map_prediction(item["response"]))

    with jsonlines.open("annotations/new_annotation.jsonl") as reader:
        instances = [obj for obj in reader]

        train_instances, dev_instances, test_instances = preprocessor.split_instances(
            instances
        )

    labels = [instance["adjudicated_label"] for instance in test_instances]

    result_folder = os.path.join(output_dir, result_foldername)
    Path(result_folder).mkdir(parents=True, exist_ok=True)

    result_filename = f"{model_name}_result"
    result_filepath = os.path.join(result_folder, result_filename)

    results = classification_report(labels, mapped_predictions, output_dict=True)

    print(classification_report(labels, mapped_predictions, digits=3))

    with open(result_filepath, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="")
    parser.add_argument("-output_dir", "--output_dir", default="./experimental_results", help="")
    parser.add_argument("-mode", "--mode", default="evaluate")
    parser.add_argument("-multiturn", "--multiturn", action="store_true")

    args = parser.parse_args()

    main(
        args.mode,
        model_name="Qwen2.5-VL-7B-Instruct",
        output_dir=args.output_dir,
        multiturn=args.multiturn,
    )