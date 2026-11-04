import argparse
import collections
import time
import os
import json
import jsonlines
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from transformers import logging
from sklearn.metrics import classification_report
from utils.learning_helper.preprocess import Preprocessor
from utils.learning_helper.learning import LearningHelper
from utils.learning_helper.evaluator import Evaluator

# control the source of randomness
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# define some global parameters
run_times = 10

# suppress all warnings for better readability
logging.set_verbosity_error()
warnings.filterwarnings('ignore')


def main(
        run_baseline,
        model,
        num_epochs,
        batch_size,
        learning_rate,
        patience,
        use_image,
        use_loc_desc,
        use_user_desc,
        fine_tune,
        pooling,
        repeated_run,
        pretrained
):

    # define the paths of annotations and tweets
    annotation_folder = './'
    tweet_folder = 'data/'

    # make sure the parameter is of right data type
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    patience = int(patience)

    # initialize the helpers
    preprocessor = Preprocessor()
    learning_helper = LearningHelper()
    evaluator = Evaluator()

    # load annotations and tweet metadata
    instances = preprocessor.load_improved_data('./improved_data.json')

    # split instances into train, development, and test
    train_instances, dev_instances, test_instances = preprocessor.split_instances(instances)

    # print the baseline results
    if run_baseline:
        test_instances = preprocessor.add_baseline_output(test_instances)
        gold_labels = [instance['adjudicated_label'] for instance in test_instances]

        print('-' * 60)
        print('Results of majority baseline')
        majority_labels = [instance['majority_baseline_output'] for instance in test_instances]
        print(classification_report(gold_labels, majority_labels))

        print('-' * 60)
        print('Results of random baseline')
        random_labels = [instance['random_baseline_output'] for instance in test_instances]
        print(classification_report(gold_labels, random_labels))

        return

    # define label mapping
    label_to_idx = {'0-24 hours': 0, '1-30 days': 1, 'more than 30 days': 2}
    idx_to_label = {0: '0-24 hours', 1: '1-30 days', 2: 'more than 30 days'}

    # check if we can use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if repeated_run:

        global_results = collections.defaultdict(list)

        model_name = learning_helper.get_model_name(
            model=model,
            use_image=use_image,
            use_loc_desc=use_loc_desc,
            use_user_desc=use_user_desc,
            pooling=pooling,
            fine_tune=fine_tune
        )
        model_name = f"{model_name}_classifier.pt"

        for run_order in range(run_times):

            print('-' * 60)
            print(f"Current batch: {run_order + 1}")
            print('-' * 60)

            classifier = learning_helper.get_model(
                model=model,
                fine_tune=fine_tune,
                use_image=use_image,
                use_user_desc=use_user_desc,
                use_loc_desc=use_loc_desc,
                pooling=pooling
            )
            classifier.to(device)

            # use pretrained model
            if pretrained:
                full_model_name = f'pretrained_{model_name}'
                model_path = os.path.join('pretrained_models', full_model_name)
                classifier.load_state_dict(torch.load(model_path))
                classifier.eval()
                break

            # retrain the model
            else:

                # define the optimizer, loss function, and some parameters

                optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
                criterion = nn.CrossEntropyLoss()

                # train the model
                best_valid_loss = float('inf')
                check_stopping = 0
                full_model_name = f'retrained_{model_name}'
                model_path = os.path.join('retrained_models', full_model_name)
                for i in range(num_epochs):

                    train_loader = preprocessor.get_data_loader(train_instances, batch_size)
                    dev_loader = preprocessor.get_data_loader(dev_instances, batch_size)

                    start_time = time.time()
                    train_loss, train_acc = learning_helper.train(classifier, train_loader, optimizer, criterion,
                                                                  device, label_to_idx)
                    dev_loss, dev_acc = learning_helper.evaluate(classifier, dev_loader, criterion, device,
                                                                 label_to_idx)
                    end_time = time.time()
                    elapsed_mins, elapsed_secs = learning_helper.epoch_time(start_time, end_time)

                    print("-" * 60)
                    print(f"Epoch: {i + 1} || Epoch Time: {elapsed_mins}m {elapsed_secs}s")
                    print(f"Epoch: {i + 1} || Train loss: {train_loss:.02f}, Train Acc: {train_acc:.02f}")
                    print(f"Epoch: {i + 1} || Dev loss: {dev_loss:.02f}, Dev Acc: {dev_acc:.02f}")

                    # check if we need to save the model
                    if dev_loss < best_valid_loss:
                        check_stopping = 0
                        best_valid_loss = dev_loss
                        torch.save(classifier.state_dict(), model_path)
                    else:
                        check_stopping += 1
                        print(f"The loss on development set does not decrease")
                        if check_stopping == patience:
                            print("The loss on development set does not decrease, stop training!")
                            break

                # evaluate the retrained/pretrained model
                test_loader = preprocessor.get_data_loader(test_instances, batch_size, shuffle=False)
                pred_labels = evaluator.test_model(classifier, test_loader, idx_to_label)
                gold_labels = [x['adjudicated_label'] for x in test_instances]
                classification_dict = evaluator.get_classification_dict(gold_labels, pred_labels)
                for metric, value in classification_dict.items():
                    global_results[metric].append(value)

        # save the evaluation results
        result_dir = 'experimental_results'
        global_statistics_filename = f'global_stats_{model_name}.json'
        global_statistics_filepath = os.path.join(result_dir, global_statistics_filename)
        with open(global_statistics_filepath, 'w') as jsonfile:
            json.dump(global_results, jsonfile)

    else:  # not maintained

        classifier = learning_helper.get_model(
            model=model,
            fine_tune=fine_tune,
            use_image=use_image,
            use_user_desc=use_user_desc,
            use_loc_desc=use_loc_desc,
            pooling=pooling
        )
        model_name = learning_helper.get_model_name(
            model=model,
            use_image=use_image,
            use_loc_desc=use_loc_desc,
            use_user_desc=use_user_desc,
            pooling=pooling,
            fine_tune=fine_tune
        )
        model_name = f"{model_name}_classifier.pt"

        # use pretrained model
        if pretrained:
            full_model_name = f'retrained_{model_name}'
            model_path = os.path.join('retrained_models', full_model_name)
            classifier.load_state_dict(torch.load(model_path))
            classifier.eval()
            with jsonlines.open('annotations/new_annotation.jsonl') as reader:
                instances = [obj for obj in reader]
                # split instances into train, development, and test
                # we only use test instances if we only do inference
                train_instances, dev_instances, test_instances = preprocessor.split_instances(instances)

        # retrain the model
        else:

            # define the optimizer, loos function, and some parameters
            optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss().to(device)

            # train the model
            best_valid_loss = float('inf')
            check_stopping = 0
            full_model_name = f'retrained_{model_name}'
            model_path = os.path.join('retrained_models', full_model_name)
            for i in range(num_epochs):

                train_loader = preprocessor.get_data_loader(train_instances, batch_size)
                dev_loader = preprocessor.get_data_loader(dev_instances, batch_size)

                start_time = time.time()
                train_loss, train_acc = learning_helper.train(classifier, train_loader, optimizer, criterion, device,
                                                              label_to_idx)
                dev_loss, dev_acc = learning_helper.evaluate(classifier, dev_loader, criterion, device, label_to_idx)
                end_time = time.time()
                elapsed_mins, elapsed_secs = learning_helper.epoch_time(start_time, end_time)

                print("-" * 60)
                print(f"Epoch: {i + 1} || Epoch Time: {elapsed_mins}m {elapsed_secs}s")
                print(f"Epoch: {i + 1} || Train loss: {train_loss:.02f}, Train Acc: {train_acc:.02f}")
                print(f"Epoch: {i + 1} || Dev loss: {dev_loss:.02f}, Dev Acc: {dev_acc:.02f}")

                # check if we need to save the model
                if dev_loss < best_valid_loss:
                    check_stopping = 0
                    best_valid_loss = dev_loss
                    torch.save(classifier.state_dict(), model_path)
                else:
                    check_stopping += 1
                    print(f"The loss on development set does not decrease")
                    if check_stopping == patience:
                        print("The loss on development set does not decrease, stop training!")
                        break

        # evaluate the retrained/pretrained model
        test_loader = preprocessor.get_data_loader(test_instances, batch_size, shuffle=False)
        pred_labels = evaluator.test_model(classifier, test_loader, idx_to_label)
        gold_labels = [x['adjudicated_label'] for x in test_instances]
        print('-' * 60)
        print(f"The classification results of {model_name}")
        print(classification_report(gold_labels, pred_labels))

        # save predictions
        evaluator.save_prediction(model_name, pred_labels, test_instances, prediction_dir='experimental_results/predictions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_baseline', '--run_baseline', action='store_true', help='')
    parser.add_argument('-model', '--model', help='')
    parser.add_argument('-num_epochs', '--num_epochs', help='')
    parser.add_argument('-batch_size', '--batch_size', help='')
    parser.add_argument('-learning_rate', '--learning_rate', help='')
    parser.add_argument('-early_stopping', '--early_stopping', help='')
    parser.add_argument('-use_image', '--use_image', action='store_true')
    parser.add_argument('-use_loc_desc', '--use_loc_desc', action='store_true')
    parser.add_argument('-use_user_desc', '--use_user_desc', action='store_true')
    parser.add_argument('-fine_tune', '--fine_tune', action='store_true')
    parser.add_argument('-pooling', '--pooling', help='')
    parser.add_argument('-repeated_run', '--repeated_run', action='store_true')
    parser.add_argument('--pretrained', action='store_true')

    args = parser.parse_args()
    main(
        run_baseline=args.run_baseline,
        model=args.model,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.early_stopping,
        use_image=args.use_image,
        use_loc_desc=args.use_loc_desc,
        use_user_desc=args.use_user_desc,
        fine_tune=args.fine_tune,
        pooling=args.pooling,
        repeated_run=args.repeated_run,
        pretrained=args.pretrained
    )
