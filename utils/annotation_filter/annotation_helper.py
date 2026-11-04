import csv
import krippendorff
import re
import os
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score

"""
USER GUIDE.
1.  annotation_list, annotators = get_annotation_list(annotation_csv, num_annotator, workers_removed=[])
    alpha = calculate_krippendorff(annotation_list)
    save_annotation_list(annotation_list, csv_filepath)

2.  If alpha < 0.6, then use MACE to discard the annotation of the worst annotator,
    and workers_removed.append(workerids).
"""


def calculate_krippendorff(annotation_list):
    temp = []
    # convert = {'Yes': 0, 'No': 1, 'Inv': 2, '': np.nan}
    convert = {'0-24 hours': 0, '1-30 days': 1, 'more than 30 days': 2, 'I cannot determine': 2, '': np.nan}
    for i in range(len(annotation_list[0])):
        temp.append([convert[x[i]] for x in annotation_list])
    res = krippendorff.alpha(reliability_data=temp)
    return res


def save_annotation_list(annotation_list, csv_filepath):
    # print("Saving the annotation list for using the MACE...")
    with open(csv_filepath, 'w') as csvfile:
        for instance in annotation_list:
            csvfile.write(','.join([ele for ele in instance]))
            csvfile.write("\n")
    # print("Done.")


def get_annotation_list(annotation_csv, workers_removed):

    annotations = defaultdict(dict)
    annotators = set()
    with open(annotation_csv, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if len(row['RequesterFeedback']) != 0:
                continue
            annotations[row['HITId']].update({row['WorkerId']: [row['Answer.q1'], row['Answer.reason']]})
            annotators.add(row['WorkerId'])

    # remove the unqualified workers
    annotators = list(annotators)
    for worker in workers_removed:
        annotators.remove(worker)

    res = []
    hitids = []
    for hitid, annotation in annotations.items():
        temp_list = []
        for annotator in annotators:

            # only deal with the Q1
            if annotator not in annotation.keys():
                temp_list.append("")
            else:
                temp_list.append(annotation[annotator][0])
        if len(temp_list) != 0:
            res.append(temp_list)
            hitids.append(hitid)
    return res, annotators, hitids
def map_prediction(model_output):
    if not model_output:
        return 'I cannot determine'

    # Use only the opening fragment (first two sentences) to decide.
    first_two = '.'.join(model_output.split('.')[:2]).strip().lower()

    if first_two.startswith('1') or 'less than 24 hours' in first_two or '0-24 hours' in first_two:
        return '0-24 hours'
    if first_two.startswith('2') or '1-30 days' in first_two:
        return '1-30 days'
    if first_two.startswith('3') or 'more than 30 days' in first_two:
        return 'more than 30 days'

    # Fallback: check the whole text for explicit phrases
    text = model_output.lower()
    if 'less than 24 hours' in text or '0-24 hours' in text:
        return '0-24 hours'
    if '1-30 days' in text:
        return '1-30 days'
    if 'more than 30 days' in text:
        return 'more than 30 days'

    return 'I cannot determine'
def get_annotation_list_LLM(annotation_csv, workers_removed):
    annotations = defaultdict(dict)
    annotators = set()
    # Loop through all files ending with "_response"
    for file_name in os.listdir(annotation_csv):
        if file_name.endswith("_response"):
            file_path = os.path.join(annotation_csv, file_name)

            # Read the content of the response file
            with open(file_path, 'r') as file:
                responses = json.load(file)

                for response_data in responses:
                    response = map_prediction(response_data.get('response', ""))
                    hit_id = response_data.get('id', "")
                    # Example to extract worker_id based on file name (assuming the file name contains worker ID)
                    model_name = file_name.split('_')[0]  # Extract model name from file name
                    annotators.add(model_name)

                    # Update the annotations with the worker's response
                    annotations[hit_id].update({model_name: response})

    # Remove unqualified workers
    annotators = list(annotators)
    for worker in workers_removed:
        if worker in annotators:
            annotators.remove(worker)

    # Prepare the result
    res = []
    hitids = []
    for hitid, annotation in annotations.items():
        temp_list = []
        for annotator in annotators:
            if annotator not in annotation.keys():
                temp_list.append("")
            else:
                temp_list.append(annotation[annotator])
        if len(temp_list) != 0:
            res.append(temp_list)
            hitids.append(hitid)
    return res, annotators, hitids

def read_competence(competence_file, annotators):
    """
    load the competence file and find the bad annotators
    :param competence_file: the competence file generated by the MACE
    :param annotators: the annotators (we need the indices of the annotators)
    :return: a list of tuples (worker_index, competence)
    """
    with open(competence_file) as file:
        competences = file.read().split("\n")[0].split("\t")
        unsorted_competences = [[float(competence), annotators[index]] for index, competence in enumerate(competences)]
        sorted_competences = sorted(unsorted_competences, key=lambda x: x[0])
    return sorted_competences


def read_prediction(prediction_filepath):
    with open(prediction_filepath, 'r') as file:
        predictions = file.read().split("\n")
    return predictions


def read_annotation_of_instance(annotation_csv, workers_removed=[], add_text_info=False, num_annotators=5):
    """
    Load the annotation file into a dictionary
    :param annotation_csv: the annotation file returned by AMT
    :return: a dictionary, key : HITid, value : annotation
    """
    
    annotations = {}
    with open(annotation_csv, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            if len(row['RequesterFeedback']) != 0:
                continue

            if row['WorkerId'] in workers_removed:
                continue

            # check if the instance information is added into dict or not
            if row['HITId'] not in annotations.keys():
                basic_info = {
                    'Input.location': row['Input.location'],
                    'Input.instance_id': row['Input.instance_id'],
                    'Input.image_link': row['Input.image_link']
                }

                # get the tweet text
                if add_text_info:
                    json_folderpath = "/home/zhaomin/Downloads/data"
                    events = os.listdir(json_folderpath)
                    event_name_tokens = re.split('(\d+)', row['Input.event'])
                    event_name = event_name_tokens[0]
                    event_year = event_name_tokens[1]
                    for event in events:
                        if event.split("_")[1] == event_name and event.split("_")[2].split("-")[0] == event_year:
                            event_folder = os.path.join(json_folderpath, event)
                            for day in os.listdir(event_folder):
                                day_folder = os.path.join(event_folder, day)
                                for x in os.listdir(day_folder):
                                    if x == row['Input.anchor_id']:
                                        anchor_jsonfilename = 'anchor_{}.json'.format(x)
                                        anchor_jsonpath = os.path.join(day_folder, x, anchor_jsonfilename)
                    with open(anchor_jsonpath, 'r') as jsonfile:
                        anchor_tweet = json.load(jsonfile)
                    tweet_text = anchor_tweet['full_text']
                    basic_info['Input.anchor_text'] = tweet_text
                annotations[row['HITId']] = basic_info

            # add the annotation to the dict
            annot = {
                'Answer.Q1_{}'.format(row['WorkerId']): row['Answer.q1'],
                'Answer.Q2_{}'.format(row['WorkerId']): row['Answer.reason']
            }
            annotations[row['HITId']].update(annot)

    return annotations


def read_annotation_of_worker(annotation_csv):
    """
    Load the annotation file into a dictionary
    :param annotation_csv: the annotation file returned by AMT
    :return: a dictionary, key : WorkerID, value : a dict (key : HITid, value : annotation)
    """
    annotations = {}
    with open(annotation_csv, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            if len(row['RequesterFeedback']) != 0:
                continue

            # check if the annotatorid is in dict or not
            if row['WorkerId'] not in annotations.keys():
                annotations[row['WorkerId']] = {}
            annot = {
                'WorkTimeInSeconds': row['WorkTimeInSeconds'],
                'Answer.Q1': row['Answer.q1'],
                'Answer.Q2': row['Answer.reason']
            }
            annotations[row['WorkerId']][row['HITId']] = annot

    return annotations


def find_crazy_annotator(annotations_of_worker, max_inv_percentage=0.9, min_seconds=5):
    """
    identify the crazy annotator
    :param annotations_of_worker: the annotation of instance (key: HITid, value: annotation)
    :param max_inv_percentage: the maximum percentage of the label INVALID
    :param min_seconds: the minimum seconds the annotators should have spent on annotation of one instance
    :return: a dictionary (key: WorkerID, value: the reason why he/she is a bad annotator)
    """
    crazy_annotators = {}
    for workerid, annots in annotations_of_worker.items():

        # check if annotator has spent more or equal to 5 seconds
        seconds_spent = [x['WorkTimeInSeconds'] for x in annots.values()]
        if not all(int(i) >= min_seconds for i in seconds_spent):
            if workerid not in crazy_annotators.keys():
                crazy_annotators[workerid] = []
            crazy_annotators[workerid].append('Less than {} seconds'.format(min_seconds))

        # check if annotator select INVALID for more than 90% instances
        num_invalid = len([x for x in annots.values() if x['Answer.Q1'] == 'Inv'])
        if num_invalid >= max_inv_percentage * len(annots):
            if workerid not in crazy_annotators.keys():
                crazy_annotators[workerid] = []
            crazy_annotators[workerid].append('Too many INVALID')

    return crazy_annotators


def find_bad_annotator(annotations_of_worker, annotations_of_instance, max_inv_percentage=0.9, min_seconds=5, min_kappa=0.4):
    """
    identify the bad annotator
    :param annotations_of_worker: the annotation of instance (key: HITid, value: annotation)
    :param max_inv_percentage: the maximum percentage of the label INVALID
    :param min_seconds: the minimum seconds the annotators should have spent on annotation of one instance
    :param min_kappa: the minimum cohen's kappa between the annotation and the majority label
    :return: a dictionary (key: WorkerID, value: the reason why he/she is a bad annotator)
    """
    bad_annotators = {}
    for workerid, annots in annotations_of_worker.items():

        # check if annotator has spent more or equal to 5 seconds
        seconds_spent = [x['WorkTimeInSeconds'] for x in annots.values()]
        if not all(int(i) >= min_seconds for i in seconds_spent):
            if workerid not in bad_annotators.keys():
                bad_annotators[workerid] = []
            bad_annotators[workerid].append('Less than {} seconds'.format(min_seconds))

        # check if annotator select INVALID for more than 90% instances
        num_invalid = len([x for x in annots.values() if x['Answer.Q1'] == 'Inv'])
        if num_invalid >= max_inv_percentage * len(annots):
            if workerid not in bad_annotators.keys():
                bad_annotators[workerid] = []
            bad_annotators[workerid].append('Too many INVALID')

        # check if the annotator has an inter-annotator agreement above 0.4 with the majority label
        q1_labels, q1_majority = [], []
        for hitid, annot in annots.items():
            q1_labels.append(annot['Answer.Q1'])
            instance = annotations_of_instance[hitid]
            instance_labels = [instance[x] for x in instance.keys() if x.startswith("Answer.Q1_")]
            majority_label = max(set(instance_labels), key=instance_labels.count)
            q1_majority.append(majority_label)
        agreement = cohen_kappa_score(q1_labels, q1_majority)
        if agreement <= min_kappa:
            if workerid not in bad_annotators.keys():
                bad_annotators[workerid] = []
            bad_annotators[workerid].append("Cohen's Kappa ({}) is too low".format(agreement))

    return bad_annotators

