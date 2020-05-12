import pickle
import argparse
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from spacy.pipeline import EntityRuler
from spacy.tokens import Span
from collections import defaultdict


# New entity labels
# Specify the new entity labels which you want to add here
LABEL = ["name", "eatType", "food", "priceRange", "customerrating", "area", "near"]

# Loading training data
def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spacy_model', help="spacy model")
    parser.add_argument('--model',help="saved model name")
    parser.add_argument('--output',help="path to output directory")
    parser.add_argument('--train', help="path to training data")
    parser.add_argument('--test', help="path to inference data")
    parser.add_argument('--iter', help="number of iterations",type=int)
    return parser.parse_args()


def train(train_data, model=None, new_model_name='new_model', output_dir=None, n_iter=10):
    """Setting up the pipeline and entity recognizer, and training the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spacy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')
    for i in LABEL:
        ner.add_label(i)   # Add new entity labels to entity recognizer

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.entity.create_optimizer()

    # Get names of other pipes to disable them during training to train only NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                           losses=losses)
            print('Losses', losses)

    # Test the trained model
    test_text = 'Gianni Infantino is the president of FIFA.'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # Save model
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

def expand_loc_entities(doc):
    new_ents = []
    for i,ent in enumerate(doc.ents):
        # Only check for title if it's a person and not the first token
        if ent.label_ == "ORDINAL":
            continue
        if ent.label_ == "LOC" and i >0:
            if doc.ents[i-1].label_ == "ORDINAL":
                new_ent = Span(doc, ent.start-1, ent.end, label=ent.label)
                new_ents.append(new_ent)
            else:
                new_ents.append(ent)
        else:
            new_ents.append(ent)
    doc.ents = new_ents
    return doc

def inference(output_dir,inference_data):
    tagged_data = defaultdict(list)
    print("Loading from", output_dir)
    # nlp2 = spacy.load('en_core_web_lg')
    nlp2 = spacy.load(output_dir)
    ruler = EntityRuler(nlp2)
    patterns = [{"label": "family-friendly", "pattern": "family - friendly"},{"label": "negate", "pattern": "not"}]
    ruler.add_patterns(patterns)
    nlp2.add_pipe(ruler, before="ner")
    for mr in inference_data:
        for test_text in inference_data[mr]:
            # Test the saved model
            doc2 = nlp2(test_text)
            tagged_txt = ""
            pad = ""
            for ent in doc2.ents:
                tagged_txt = tagged_txt + "{}{}:{}".format(pad,ent.label_, ent.text)
                pad = ","
            tagged_data[mr].append(tagged_txt)
                # print(ent.label_, ent.text,end=",")
            # print("")
    return tagged_data

def get_inference_data(test_file=None):
    if test_file:
        with open(test_file,"r") as f:
            test_data = [line.rstrip() for line in f]
    else:
        test_data = ["There is a place in the city centre, Alimentum, that is not family-friendly."]
    return test_data


def get_train_data(train_file=None):
    if train_file:
        with open(train_file, 'rb') as fp:
            train_data = pickle.load(fp)
    else:
        with open('./entity-annotated-corpus/ner_corpus_260.spacy', 'rb') as fp:
            train_data = pickle.load(fp)
    return train_data


# def main():
#     args = parse_argument()
#     output_dir = Path("/Users/yfeng/Public/Study/20Spring/11727/project/neural-template-gen/e2e_ner")
#     test_data = "/Users/yfeng/Public/Study/20Spring/11727/project/neural-template-gen/e2e_example/presentation"
#     inference_data = get_inference_data(test_data+".txt")
#     inference(output_dir, inference_data,test_data+".tagged.txt")
#
# if __name__ == '__main__':
#     main()