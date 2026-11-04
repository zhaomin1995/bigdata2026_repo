import os
import torch
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class Evaluator:

    def __init__(self):
        pass

    def test_model(self, model, test_loader, idx_to_label):
        """Get the prediction using the trained model and test data.

        Args:
            model: trained model
            test_loader: the data loader of test data
            idx_to_label: convert the index to the label
        Returns:
            The predicted labels
        """

        # set the model as evaluation mode
        model.eval()

        pred_labels = []
        for instances in test_loader:
            with torch.no_grad():
                preds = model(instances)
                pred_ids = preds.argmax(dim=1)
                for pred_id in pred_ids:
                    pred_label = idx_to_label[pred_id.item()]
                    pred_labels.append(pred_label)

        return pred_labels

    def get_classification_dict(self, gold_labels, pred_labels):
        """Get the detailed metrics of each class.

        :param gold_labels: the ground-truth
        :param pred_labels:  the predicted label
        :return:
        """
        # get the precision, recall, and F1 for each class
        precisions = precision_recall_fscore_support(gold_labels, pred_labels, average=None,
                                                     labels=['0-24 hours', '1-30 days', 'more than 30 days'])[0]
        recalls = precision_recall_fscore_support(gold_labels, pred_labels, average=None,
                                                  labels=['0-24 hours', '1-30 days', 'more than 30 days'])[1]
        f1s = precision_recall_fscore_support(gold_labels, pred_labels, average=None,
                                              labels=['0-24 hours', '1-30 days', 'more than 30 days'])[2]
        hour_precision, day_precision, more_precision = np.round(precisions, 2)
        hour_recall, day_recall, more_recall = np.round(recalls, 2)
        hour_f1, day_f1, more_f1 = np.round(f1s, 2)

        # get the weighted precision, recall, and F1 for all data
        weighted_metrics = precision_recall_fscore_support(gold_labels, pred_labels, average='weighted')
        weighted_metrics = np.round(weighted_metrics[:3], 2)
        average_precision, average_recall, average_f1 = weighted_metrics

        result = {
            'hour_precision': hour_precision,
            'hour_recall': hour_recall,
            'hour_f1': hour_f1,
            'day_precision': day_precision,
            'day_recall': day_recall,
            'day_f1': day_f1,
            'more_precision': more_precision,
            'more_recall': more_recall,
            'more_f1': more_f1,
            'average_precision': average_precision,
            'average_recall': average_recall,
            'average_f1': average_f1
        }

        return result

    def save_prediction(self, model_name, pred_labels, test_instances, prediction_dir='predictions'):
        """Save the predicted labels to do the error analysis.

        Args:
            model_name: text_only OR image_only OR text_image
            pred_labels: predicted labels returned by train model
            test_instances: test instances
            prediction_dir:
        Returns:
            None
        """
        # save the predicted label
        prediction_filename = f'prediction_{model_name}.json'
        prediction_filepath = os.path.join(prediction_dir, prediction_filename)
        with open(prediction_filepath, 'w+') as pred_file:
            for index, instance in enumerate(test_instances):

                # add the predicted label
                pred_label = pred_labels[index]
                instance['pred_label'] = pred_label

                # save the instance
                pred_file.write(json.dumps(instance))
                pred_file.write('\n')
