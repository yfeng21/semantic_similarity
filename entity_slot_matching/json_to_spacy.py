# Convert json file to spaCy format.
import plac
import logging
import argparse
import sys
import os
import json
import pickle

def main(input_file=None, output_file=None):
    try:
        training_data = []
        lines=[]
        with open(input_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    entities.append((point['start'], point['end'] + 1 ,label))


            training_data.append((text, {"entities" : entities}))

        # print(training_data)

        with open(output_file, 'wb') as fp:
            pickle.dump(training_data, fp)

    except Exception as e:
        logging.exception("Unable to process " + input_file + "\n" + "error = " + str(e))
        return None
if __name__ == '__main__':
    #data_dir = "/Users/yfeng/Public/Study/20Spring/11727/project/neural-template-gen/e2e_example/valid"
    data_dir = "/home/yulan/direction_nlg/template_generation/neural-template-gen/data/e2e_aligned/train"
    main(data_dir + '.json', data_dir + '.spacy')
