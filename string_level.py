from itertools import combinations
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import matplotlib.pyplot as plt
import re
from ast import literal_eval
import csv
from collections import defaultdict
import editdistance

def extract_slots_pair(sent):
    if ":" in sent:
        pairs = [(s.split(":")[0],s.split(":")[1]) for s in sent.rstrip().split(",")]
    else:
        pairs = []
        slots = sent.rstrip().split(",")
        for s in slots:
            m = re.search(r"\[(.*?)\]", s)
            pairs.append((s.split("[")[0],m.group(1)))
    return pairs


def evaluate_slot_sim(sent1,sent2):
    pairs1 = extract_slots_pair(sent1)
    slots1, values1 = zip(*pairs1)
    pairs2 = extract_slots_pair(sent2)
    slots2, values2 = zip(*pairs2)
    common_slots = set(slots1) & set(slots2)
    common_values = set(pairs1) & set(pairs2)
    sim = (len(common_slots)+len(common_values))/(len(pairs1)+len(pairs2))
    return sim

def evaluate_bleu_sim(hype, ref):
    score = sentence_bleu(ref, hype,weights=(0.5,0.5))
    return score


def evaluate_edit_sim(sent1,sent2):
    distance = editdistance.eval(sent1.split(), sent2.split())
    sum_len = len(sent1.split())+len(sent2.split())
    score = (sum_len-distance)/sum_len
    return score

def evaluate_in_group(sents,mode):
    scores = []
    if mode == "slot" or mode == "edit":
        x = [i for i in range(len(sents))]
        pairs = list(combinations(x, 2))
        for p in pairs:
            if mode == "slot":
                scores.append(evaluate_slot_sim(sents[p[0]],sents[p[1]]))
            else:
                scores.append(evaluate_edit_sim(sents[p[0]], sents[p[1]]))
    elif mode == "bleu":
        for i in sents:
            ref = sents[:]
            ref.remove(i)
            score = evaluate_bleu_sim(i, ref)
            scores.append(score)
    scores = np.array(scores)
    mean = np.mean(scores)
    var = np.var(scores)
    # print("in group similarirty:{:.2f}%, var:{:.2f}".format(mean*100,var))
    return mean,var


def evaluate_between_group(group1,group2,mode):
    score = 0
    total = 0
    for i in group1:
        for j in group2:
            if mode == "slot":
                score += evaluate_slot_sim(i, j)
                total += 1
            elif mode == "bleu":
                score += evaluate_bleu_sim(i,group2)
                score += evaluate_bleu_sim(j, group1)
                total += 2
    score /= total
    print("intra group similarirty:{:.2f}%  ".format(score*100))


def read_into_groups(file_in):
    all_sents = []
    sents = []
    with open(file_in,"r") as f:
        for line in f:
            if line != "\n":
                sents.append(line)
            else:
                all_sents.append(sents)
                sents = []
    all_sents.append(sents)
    return all_sents

def evaluate_slot(all_groups):
    evaluate_in_group(all_groups[0],"slot")
    evaluate_between_group(all_groups[0], all_groups[1],"slot")
    evaluate_between_group(all_groups[0], all_groups[2],"slot")

def evaluate_bleu(all_groups_dict):
    in_group_bleu = []
    count_1 = 0
    for group in all_groups_dict:
        if len(all_groups_dict[group]) < 2:
            count_1 += 1
            continue
        score,var = evaluate_in_group(all_groups_dict[group],"bleu")
        in_group_bleu.append({"mr":group, "bleu score":score,"bleu variance":var})
    print("{} groups not included because they have less than 2 sentences".format(count_1))
    return in_group_bleu
    # evaluate_between_group(all_groups[0], all_groups[1],"bleu")
    # evaluate_between_group(all_groups[0], all_groups[2],"bleu")


def test_func():
    sent1 = "here is a place in the city centre , Alimentum , that is not family - friendly ."
    sent2 = "In the city centre there is a venue name Alimentum , this is not a family - friendly venue "
    evaluate_bleu_sim(sent1, [sent2])
    evaluate_bleu_sim(sent2, [sent1])

def heatmap(x_labels, y_labels, values):
    fig, ax = plt.subplots()
    im = ax.imshow(values)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10,
         rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, "%.2f"%values[i, j],
                           ha="center", va="center", color="w",
fontsize=6)
    fig.tight_layout()
    plt.show()

def corr_sim(corr):
    score = 0
    for i in range(len(corr)):
        score += (sum(corr[i])-corr[i][i])/(len(corr)-1)
    score /= len(corr)
    print("intra group similarirty:{:.2f}%  ".format(score*100))

def plot_heat_map(corr_txt):
    corr = re.sub('\s+', ',', corr_txt)
    corr = np.array(literal_eval(corr))
    # x = [i for i in range(6)]
    # corr_sim(corr)
    # print(corr[:6][:6])
    x = [0,1,6,7,11,]
    heatmap(x, x, corr)

def evaluate_gold():
    sent1 = "name[Alimentum], area[city centre], familyFriendly[yes]"
    sent2 = "name[Aromi], eatType[coffee shop], food[Chinese], customer rating[average], area[city centre], familyFriendly[yes]"
    evaluate_slot_sim(sent1, sent2)


def read_e2e_csv(test_csv):
    groups_set = defaultdict(set)
    groups = defaultdict(list)
    count = 0
    with open(test_csv+".csv","r") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            groups_set[row["mr"]].add(row["ref"])
            count += 1
    print("{} mr groups of {} nl sentences".format(len(groups_set),count))
    for g in groups_set:
        groups[g] = list(groups_set[g])
    return groups


def write_csv(result_dict,test_csv,metric):
    fields = ["mr",metric+" score",metric+" variance"]
    with open(test_csv+metric+".csv","w") as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=fields)
        writer.writeheader()
        for r in result_dict:
            # {"score": score, "variance": var}
            writer.writerow(r)


def write_bleu(test_data, test_out):
    in_group_bleu = evaluate_bleu(test_data)
    write_csv(in_group_bleu, test_out, "bleu")


def evaluate_edit_distance(all_groups_dict):
    in_group_edit = []
    count_1 = 0
    for group in all_groups_dict:
        if len(all_groups_dict[group]) < 2:
            count_1 += 1
            continue
        score, var = evaluate_in_group(all_groups_dict[group], "edit")
        in_group_edit.append({"mr": group, "edit score": score, "edit variance": var})
    print("{} groups not included because they have less than 2 sentences".format(count_1))
    return in_group_edit


def write_edit_distance(test_data, test_out):
    in_group_edit = evaluate_edit_distance(test_data)
    write_csv(in_group_edit, test_out, "edit")


if __name__ == '__main__':
    data_dir = "/Users/yfeng/Public/Study/20Spring/11727/project/e2e-cleaning/cleaned-data/"
    out_dir = "/Users/yfeng/Public/Study/20Spring/11727/project/measure_similarity/test_out/"
    test_csv = data_dir+"test-fixed"
    dev_csv = data_dir+"devel-fixed.no-ol"
    test_out = out_dir + "test-"
    dev_out = out_dir + "dev-"
    test_data = read_e2e_csv(test_csv)
    dev_data = read_e2e_csv(dev_csv)
    write_bleu(test_data,test_out)
    write_bleu(dev_data, dev_out)
    # write_edit_distance(test_data,test_out)
    # tagged_file = "/Users/yfeng/Public/Study/20Spring/11727/project/neural-template-gen/e2e_example/presentation.tagged.txt"
    # all_groups = read_into_groups(file_in)
    # plot_heat_map()
    # evaluate_slot(all_groups)

    # evaluate_gold()


