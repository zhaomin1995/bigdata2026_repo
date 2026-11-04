import csv
import numpy as np
from collections import defaultdict


class BadAnnotatorIdentifier:

    def __init__(self):
        """Define the global parameters"""
        self.minimum_work_time = 5

    def load_annotator(self, batch_filepath):
        """Load annotations based on annotators.

        Args:
            The path of the batch file downloaded on AMT.
        Returns:
            A dictionary as {worker_id: {HITID: instance_metadata}}
        """
        worker_info = defaultdict(dict)
        with open(batch_filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for instance in reader:
                worker_id = instance['WorkerId']
                hit_id = instance['HITId']
                worker_info[worker_id][hit_id] = instance
        return worker_info

    def identify_bad_annotator(self, batch_filepath):
        """Determine if the annotator is bad or not.

        Args:
            The path of the batch file downloaded on AMT.
        Returns:
            A dictionary as {worker_id: the reason why he/she is bad}
        """
        annotator_info = self.load_annotator(batch_filepath)
        bad_annotator = {}
        for worker_id, annotations in annotator_info.items():

            # any workers that spend less than 5 seconds in average (avg of 3 percentiles) is bad annotator
            work_times = [int(annot['WorkTimeInSeconds']) for annot in annotations.values()]
            percentile_25 = np.percentile(work_times, 25)
            percentile_50 = np.percentile(work_times, 50)
            percentile_75 = np.percentile(work_times, 75)
            work_time_average = (percentile_25 + percentile_50 + percentile_75) / 3
            if work_time_average < self.minimum_work_time:
                bad_annotator[worker_id] = 'work time is not enough'
            
        return bad_annotator
