import json
import os
import torch
import random
import numpy as np

from collections import Counter
from torchvision import transforms
from tqdm import tqdm


class Preprocessor:

    def __init__(self):
        # define global parameters
        pass

    def load_annotations(self, annotation_folder, tweet_loader, load_silver=False):
        """Load annotations and tweets' metadata.

        Args:
            annotation_folder: the path to the folder of annotation
            tweet_loader: the path to the folder of tweets (include metadata ana image files)
            load_silver: whether it loads silver annotations
        Return:
            A list of dictionaries containing metadata and annotations
        """
        instances = []
        annotation_filenames = os.listdir(annotation_folder)
        for annotation_filename in annotation_filenames:

            # load annotations batch by batch
            annotation_filepath = os.path.join(annotation_folder, annotation_filename)
            with open(annotation_filepath, 'r') as annotation_file:
                lines = annotation_file.read().split('\n')[:-1]
                for line in lines:
                    annot = json.loads(line)
                    instance_id = annot['instance_id']

                    # skip the instances annotated as "I cannot determine"
                    if annot['adjudicated_label'] == 'I cannot determine':
                        continue

                    # skip silver annotations if needed
                    if not load_silver and annot['annotation_type'] == 'silver':
                        continue

                    # use instance_id to locate tweet metadata
                    tweet_folder_path = os.path.join(tweet_loader, instance_id)
                    metadata_filename = f'{instance_id}.json'
                    metadata_filepath = os.path.join(tweet_folder_path, metadata_filename)
                    image_filename = f'{instance_id}.png'
                    image_filepath = os.path.join(tweet_folder_path, image_filename)

                    # load metadata
                    with open(metadata_filepath, 'r') as metadata_file:
                        metadata = json.load(metadata_file)

                    # add metadata into the loaded annotation
                    annot.update({'image_filepath': image_filepath})
                    annot.update({'metadata': metadata})
                    instances.append(annot)

        # shuffle the instances
        random.shuffle(instances)

        return instances

    def load_improved_data(self, data_filename):
        with open(data_filename, 'r', encoding='utf-8') as file:
            instances = json.load(file)
        return instances

    def add_baseline_output(self, instances):
        """Add baseline outputs to the instances.

        Args:
            instances: a list of dictionaries without baseline outputs
        Returns:
            A list of dictionaries containing baseline outputs
        """
        all_labels = [x['adjudicated_label'] for x in instances]
        label_count = Counter(all_labels)

        # get the majority label
        majority_label = label_count.most_common()[0][0]

        # get the unique labels
        labels = list(set(all_labels))

        # add baseline outputs to the instances
        for instance in instances:
            random_label = random.choices(labels)[0]
            instance['majority_baseline_output'] = majority_label
            instance['random_baseline_output'] = random_label

        return instances

    def split_instances(self, instances):
        """Split data into training data, development data, and test data.

        Args:
            instances: data to be split
        Returns:
            training data, development data, and test data
        """
        data_split_filepath = 'data_split'
        with open(data_split_filepath, 'r') as split_file:
            splits = json.load(split_file)
            train_ids = splits['train']
            dev_ids = splits['dev']
            test_ids = splits['test']
        training_instances = [instance for instance in instances if instance['instance_id'] in train_ids]
        development_instances = [instance for instance in instances if instance['instance_id'] in dev_ids]
        test_instances = [instance for instance in instances if instance['instance_id'] in test_ids]
        test_instances = sorted(test_instances, key=lambda x: x['instance_id'])  # fix the order of test data
        return training_instances, development_instances, test_instances

    def get_data_loader(self, instances, batch_size, shuffle=True):
        """
        Get the data loader of this model
        :param instances: the instances in each batch
        :param batch_size: the predefined batch size
        :param shuffle: whether we need to shuffle the instances in each batch
        :return: data loader (generator)
        """
        instances_size = len(instances)
        for index in range(0, instances_size, batch_size):
            batch = instances[index: min(index + batch_size, instances_size)]
            if shuffle:
                random.shuffle(batch)
            yield batch

