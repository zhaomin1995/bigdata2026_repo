import argparse
import json
import os
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report

from utils.annotation_filter.annotation_helper import map_prediction
from utils.learning_helper.preprocess import Preprocessor


LABELS = [
    "0-24 hours",
    "1-30 days",
    "more than 30 days",
    "I cannot determine",
]


def load_test_instances(annotation_file="annotations2/new_annotation.jsonl"):
    preprocessor = Preprocessor()

    with open(annotation_file, "r", encoding="utf-8") as file:
        instances = [json.loads(line) for line in file if line.strip()]

    _, _, test_instances = preprocessor.split_instances(instances)
    gold_by_id = {
        instance["instance_id"]: instance["adjudicated_label"]
        for instance in test_instances
    }
    return test_instances, gold_by_id


def load_response_file(response_filepath):
    with open(response_filepath, "r", encoding="utf-8") as file:
        raw_payload = json.load(file)

    if isinstance(raw_payload, dict):
        raw_payload = [raw_payload]

    parsed_entries = []
    for item in raw_payload:
        if isinstance(item, dict):
            parsed_entries.append(
                {
                    "id": item.get("id"),
                    "prediction": map_prediction(item.get("response", "")),
                }
            )
        else:
            parsed_entries.append(
                {
                    "id": None,
                    "prediction": map_prediction(str(item)),
                }
            )

    return parsed_entries


def score_response_file(response_filepath, test_instances, gold_by_id):
    parsed_entries = load_response_file(response_filepath)
    has_ids = any(entry["id"] is not None for entry in parsed_entries)

    gold_labels = []
    predictions = []
    missing_ids = 0

    if has_ids:
        for entry in parsed_entries:
            instance_id = entry["id"]
            if instance_id in gold_by_id:
                gold_labels.append(gold_by_id[instance_id])
                predictions.append(entry["prediction"])
            else:
                missing_ids += 1
    else:
        for instance, entry in zip(test_instances, parsed_entries):
            gold_labels.append(instance["adjudicated_label"])
            predictions.append(entry["prediction"])

    if not gold_labels:
        raise ValueError(f"No comparable predictions found in {response_filepath}")

    report = classification_report(
        gold_labels,
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        gold_labels,
        predictions,
        labels=LABELS,
        digits=3,
        zero_division=0,
    )
    accuracy = accuracy_score(gold_labels, predictions)

    return {
        "file": os.path.basename(response_filepath),
        "count": len(predictions),
        "accuracy": accuracy,
        "missing_ids": missing_ids,
        "report": report,
        "report_text": report_text,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate every saved response file in one command."
    )
    parser.add_argument(
        "--responses_dir",
        default="experimental_results/responses",
        help="Folder containing *_response files.",
    )
    parser.add_argument(
        "--annotation_file",
        default="annotations2/new_annotation.jsonl",
        help="Ground-truth annotations used to build the test split.",
    )
    parser.add_argument(
        "--summary_file",
        default="experimental_results/results/response_test_summary.json",
        help="JSON summary output path.",
    )
    args = parser.parse_args()

    responses_dir = Path(args.responses_dir)
    if not responses_dir.exists():
        raise FileNotFoundError(f"Response directory not found: {responses_dir}")

    response_files = sorted(
        path
        for path in responses_dir.iterdir()
        if path.is_file() and path.name.endswith("_response")
    )
    if not response_files:
        raise FileNotFoundError(f"No *_response files found in {responses_dir}")

    test_instances, gold_by_id = load_test_instances(args.annotation_file)

    summaries = []
    for response_file in response_files:
        summary = score_response_file(str(response_file), test_instances, gold_by_id)
        summaries.append(summary)

        print(f"\n=== {summary['file']} ===")
        print(f"Samples evaluated: {summary['count']}")
        if summary["missing_ids"]:
            print(f"Missing ids skipped: {summary['missing_ids']}")
        print(f"Accuracy: {summary['accuracy']:.3f}")
        print(summary["report_text"])

    Path(args.summary_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_file, "w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)


if __name__ == "__main__":
    main()