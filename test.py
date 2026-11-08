import os
import csv
from matplotlib import font_manager
from matplotlib.animation import adjusted_figsize
import numpy as np
import matplotlib.pyplot as plt
from utils.annotation_filter.annotation_helper import *
from textwrap import wrap
from collections import Counter
import subprocess

from utils.learning_helper.preprocess import Preprocessor
# generate the annotation file with new replaced labels

annotation_csv = 'experimental_results/responses'

workers_removed = []
k_threshold = 0.6
k_alpha = 0.0
loc_tmp_dir = '.'
while k_alpha < k_threshold and (len(workers_removed) < len(os.listdir(annotation_csv)) - 2):

    # generate the file to use MACE, the generated file is 'annotation_list.csv'
    annotation_list_csvfilepath = 'annotation_list.csv'
    annotation_list, annotators, hitids = get_annotation_list_LLM(annotation_csv, workers_removed)
    save_annotation_list(annotation_list, annotation_list_csvfilepath)

    # run MACE
    mace_prefix = "LLM"
    subprocess.run(
        f"java -jar other/MACE.jar --prefix {mace_prefix} --test gt.key annotation_list.csv >/dev/null 2>&1",
        shell=True,
    )

    # load the competence file and prediction file
    competence_file = f'{loc_tmp_dir}/{mace_prefix}.competence'
    competence_annotators = read_competence(competence_file, annotators)
    print(competence_annotators)
    mace_prediction_file = f'{loc_tmp_dir}/{mace_prefix}.prediction'
    predictions = read_prediction(mace_prediction_file)

    raw_preds = {}
    for index, hitid in enumerate(hitids):
        pred = predictions[index]
        raw_preds[hitid] = pred

    max_num_instance = len([x[0] for x in annotation_list])
    workers_removed.append(competence_annotators[0][1])
    annotation_list, annotators, hitids = get_annotation_list_LLM(annotation_csv, workers_removed)
    # calculate Krippendorf's Alpha
    k_alpha = calculate_krippendorff(annotation_list)
    if k_alpha >= k_threshold or (len(workers_removed) >= len(os.listdir(annotation_csv)) - 2):
        print(f"The overall k_alpha is {k_alpha:.2f}, you have removed {len(workers_removed)} annotators")
        for idx, x in enumerate(annotation_list):
            for i in range(len(annotation_list[0])):
                if i> 0 and x[i]!= x[i-1]:
                    print(hitids[idx], x, annotators)
                    break
        break
    else:
        print(f"Current k_alpha: {k_alpha:.2f}, you have removed {len(workers_removed)} annotators")

json_filename = 'new_annotation.jsonl'
annotation_folder = 'annotations2/'
json_filepath = os.path.join(annotation_folder, json_filename)
preprocessor = Preprocessor()
instances = preprocessor.load_improved_data('improved_data.json')

# split instances into train, development, and test
# we only use test instances if we only do inference
train_instances, dev_instances, test_instances = preprocessor.split_instances(instances)
with open(json_filepath, 'w') as jsonfile:
    for index, hitid in enumerate(hitids):
        instance  = {
            'instance_id': hitid,
            'adjudicated_label': predictions[index],
            'metadata':{'full_text':test_instances[index]['metadata']['full_text']},
            'image_filepath': test_instances[index]['image_filepath']
        }
        jsonfile.write(json.dumps(instance))
        jsonfile.write('\n')



# manually generate pie chart
fig, ax = plt.subplots(figsize=(10, 8))
fontsize = 28
font = font_manager.FontProperties(family='Comic Sans MS', weight='medium', style='normal', size=fontsize)
pie_chart_labels = ['0-24 hours', '1-30 days', 'more than 30 days', 'I cannot determine']
group =  {x:predictions.count(x) for x in predictions}
pie_chart_data = [group['0-24 hours'], group['1-30 days'], group['more than 30 days'], group['I cannot determine']]
ax.pie(pie_chart_data, labels=pie_chart_labels, autopct='%.1f%%', textprops={'fontsize': fontsize})
# Adjust legend position
ax.legend(loc="center left", bbox_to_anchor=(1, 0.2), prop=font)
ax.axis('equal')
plt.savefig('label_dis.jpg',dpi=500, bbox_inches="tight")
plt.savefig('label_dis.pdf',dpi=500, bbox_inches="tight")