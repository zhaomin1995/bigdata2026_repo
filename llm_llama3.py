import os
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/DATA/hf_cache/"

import argparse
import json
import torch
import jsonlines
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.learning_helper.preprocess import Preprocessor
from utils.annotation_filter.annotation_helper import map_prediction

def clean_thinking_tokens(raw_text: str) -> str:
    if "</think>" in raw_text:
        return raw_text.split("</think>")[-1].strip()
    return raw_text.strip()

def llama_interface(args, tokenizer, model):
    if "messages" in args:
        messages = args["messages"]
    else:
        messages = [
            {"role": "user", "content": args["query"]},
        ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2100,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    outputs = tokenizer.batch_decode(
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

        model_path = "meta-llama/Llama-3.2-3B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
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
                "1. 0-24 hours.\n"
                "2. 1-30 days.\n"
                "3. more than 30 days.\n\n"
                "Please give your answer in the following format: Option: Reason\n"
            )

            if multiturn:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant for inferring the duration of human activities."},
                    {"role": "user", "content": f"Analyze how the tweet's text and hashtags relate to the location.\n\nTweet text: {tweet_text}"}
                ]
                llama_output_turn1 = llama_interface({"messages": messages}, tokenizer, model)
                messages.append({"role": "assistant", "content": llama_output_turn1})
                messages.append({"role": "user", "content": prompt})
                llama_output = llama_interface({"messages": messages}, tokenizer, model)
            else:
                args_dict = {
                    "query": prompt,
                }
                llama_output = llama_interface(args_dict, tokenizer, model)

            responses.append(
                {
                    "response": llama_output,
                    "id": instance["instance_id"],
                }
            )

            mapped_output = map_prediction(llama_output)
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

    result_filename = f"{model_name}_{'multiturn_' if multiturn else ''}result"
    result_filepath = os.path.join(result_folder, result_filename)

    results = classification_report(labels, mapped_predictions, output_dict=True)

    print(classification_report(labels, mapped_predictions, digits=3))

    with open(result_filepath, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model", default="meta-llama/Llama-3.2-3B-Instruct", help="")
    parser.add_argument("-output_dir", "--output_dir", default="./experimental_results", help="")
    parser.add_argument("-mode", "--mode", default="evaluate")
    parser.add_argument("-multiturn", "--multiturn", action="store_true")

    args = parser.parse_args()

    main(
        args.mode,
        model_name="Llama3.2",
        output_dir=args.output_dir,
        multiturn=args.multiturn,
    )
