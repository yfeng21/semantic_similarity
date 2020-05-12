from itertools import combinations
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import matplotlib.pyplot as plt
import re
from ast import literal_eval
import csv
import editdistance
from entity_slot_matching.train_e2e_ner import *
import os.path


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


def evaluate_gold(pairs):
    count = 1
    for p in pairs:
        sent1 = p[1]
        sent2 = p[2]
        print(count,end=":")
        evaluate_slot_sim(sent1, sent2)
        count += 1


def extract_slots_pair(sent):
    try:
        if ":" in sent:
            pairs = [(s.split(":")[0],s.split(":")[1]) for s in sent.rstrip().split(",")]
        else:
            pairs = []
            slots = sent.rstrip().split(",")
            for s in slots:
                m = re.search(r"\[(.*?)\]", s)
                pairs.append((s.split("[")[0].strip(),m.group(1)))
    except:
        pairs = [("","")]
    return pairs


def evaluate_slot_sim(sent1,sent2):
    pairs1 = extract_slots_pair(sent1)
    slots1, values1 = zip(*pairs1)
    pairs2 = extract_slots_pair(sent2)
    slots2, values2 = zip(*pairs2)
    common_slots = set(slots1) & set(slots2)
    common_values = set(pairs1) & set(pairs2)
    sim = (len(common_slots)+len(common_values))/(len(pairs1)+len(pairs2))
    print(sim*12)
    return sim


def write_into_groups(test_data, test_out):
    with open(test_out,"w") as f:
        for mr in test_data:
            f.write("MR:{}\n".format(mr))
            for nl in test_data[mr]:
                f.write(nl+"\n")


def extract_slots(test_data,test_out):
    trained_model = Path("./entity_slot_matching/e2e_ner")
    tagged_data = inference(trained_model, test_data)
    write_into_groups(tagged_data, test_out+".tagged.txt")
    return tagged_data

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


def evaluate_between_group(test_data,pair,mode):
    # pair:[notes,mr1,mr2]
    notes = pair[0]
    mr1 = pair[1]
    mr2 = pair[2]
    scores = []
    # ["mr1", "nl1", "mr2", "nl2", mode + " score"]
    for i in test_data[mr1]:
        for j in test_data[mr2]:
            if mode == "slot":
                score = evaluate_slot_sim(i, j)
            elif mode == "edit":
                score = evaluate_edit_sim(i, j)
            elif mode == "bleu":
                score = evaluate_bleu_sim(i,test_data[mr2])
                score += evaluate_bleu_sim(j, test_data[mr1])
                score /= 2
            scores.append({"notes":notes,"mr1":mr1, "nl1":i, "mr2":mr2, "nl2":j, mode + " score":score})
    return scores

def read_into_groups(file_in):
    all_sents_by_mr = defaultdict(list)
    sents = []
    mr = ""
    with open(file_in,"r") as f:
        for line in f:
            if not line.startswith("MR"):
                sents.append(line.rstrip())
            else:
                if mr != "":
                    all_sents_by_mr[mr]= sents
                    sents = []
                mr = line.rstrip().split(":")[1]
    all_sents_by_mr[mr]= sents
    return all_sents_by_mr


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


def write_csv(result_dict,test_csv,metric,ingroup=True):
    if ingroup:
        fields = ["mr",metric+" score",metric+" variance"]
    else:
        fields = ["notes","mr1","nl1","mr2","nl2",metric+" score"]
    with open(test_csv+metric+".csv","w") as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=fields)
        writer.writeheader()
        for r in result_dict:
            writer.writerow(r)


def get_ingroup_scores(all_groups_dict,metric):
    print(metric)
    in_group_scores = []
    count_1 = 0
    for group in all_groups_dict:
        if len(all_groups_dict[group]) < 2:
            count_1 += 1
            continue
        score, var = evaluate_in_group(all_groups_dict[group], metric)
        in_group_scores.append({"mr": group, metric+" score": score, metric+" variance": var})
    print("{} groups not included because they have less than 2 sentences".format(count_1))
    return in_group_scores


def write_edit_distance(test_data, test_out):
    in_group_edit = get_ingroup_scores(test_data,"edit")
    write_csv(in_group_edit, test_out, "edit")



def write_bleu(test_data, test_out):
    in_group_bleu = get_ingroup_scores(test_data,"bleu")
    write_csv(in_group_bleu, test_out, "bleu")


def write_slot(test_data, test_out):
    if os.path.exists(test_out+"nl-by-group.tagged.txt"):
        tagged_data = read_into_groups(test_out+"nl-by-group.tagged.txt")
    else:
        tagged_data = extract_slots(test_data, test_out + "nl-by-group")
    in_group_slot = get_ingroup_scores(tagged_data,"slot")
    write_csv(in_group_slot, test_out, "slot")


def run_subset_ingroup(test_csv, test_out):
    test_data = read_e2e_csv(test_csv) #{MR:[NL]}
    # write_bleu(test_data,test_out)
    # write_edit_distance(test_data,test_out)
    write_slot(test_data,test_out)


def read_mr_pair_file(mr_pair_file):
    # return a list of lists
    mr_pairs = []
    with open(mr_pair_file,"r") as f:
        for line in f:
            mr_pairs.append(line.rstrip().split("|"))
    return mr_pairs

def write_betweengroup_bleu(test_data,mr_pairs,test_out):
    between_group_results = []
    for p in mr_pairs:
        between_group_results.extend(evaluate_between_group(test_data, p, "bleu"))
    write_csv(between_group_results, test_out, "bleu",ingroup=False)

def write_betweengroup_edit(test_data,mr_pairs,test_out):
    between_group_results = []
    for p in mr_pairs:
        between_group_results.extend(evaluate_between_group(test_data, p, "edit"))
    write_csv(between_group_results, test_out, "edit",ingroup=False)


def write_betweengroup_slot(test_data,mr_pairs,test_out):
    if os.path.exists(test_out+"nl-by-group.tagged.txt"):
        tagged_data = read_into_groups(test_out+"nl-by-group.tagged.txt")
    else:
        tagged_data = extract_slots(test_data, test_out + "nl-by-group")
    between_group_results = []
    for p in mr_pairs:
        between_group_results.extend(evaluate_between_group(tagged_data, p, "slot"))
    write_csv(between_group_results, test_out, "slot", ingroup=False)

def run_subset_between_group(test_csv, test_out, mr_pair_file):
    test_data = read_e2e_csv(test_csv)
    mr_pairs = read_mr_pair_file(mr_pair_file) # [notes,mr1,mr2]
    # write_betweengroup_bleu(test_data,mr_pairs,test_out)
    # write_betweengroup_edit(test_data, mr_pairs, test_out)
    write_betweengroup_slot(test_data, mr_pairs, test_out)


def read_and_eval_negation(negation_csv, test_out):
    test_data = defaultdict(list)
    mr_pairs = []
    with open(negation_csv,"r") as f:
        csv_reader = csv.reader(f,dialect="excel-tab")
        neg_pair = set()
        for row in csv_reader:
            if row[0] != "":
                neg_pair.add(row[0])
                test_data[row[0]].append(row[1])
            elif len(neg_pair) == 2:
                mr_pairs.append(["negation pair"]+list(neg_pair))
                neg_pair = set()
    mr_pairs.append(["negation pair"]+list(neg_pair))
    write_betweengroup_bleu(test_data,mr_pairs,test_out)
    write_betweengroup_edit(test_data, mr_pairs, test_out)
    write_betweengroup_slot(test_data, mr_pairs, test_out)


def read_and_eval_corr(corr_csv, test_out):
    test_data = defaultdict(list)
    pair_by_num = defaultdict(set)
    mr_pairs = []
    with open(corr_csv, "r") as f:
        csv_reader = csv.reader(f,quotechar='"', delimiter=',',
                     quoting=csv.QUOTE_ALL, skipinitialspace=True)
        for row in csv_reader:
            if row[0] != "":
                nums = row[2].split(',')
                for num in nums:
                    pair_by_num[num.split("_")[0]].add(row[0])
                test_data[row[0]].append(row[1])
    for num in pair_by_num:
        mr_pairs.append([num] + list(pair_by_num[num]))
    write_betweengroup_bleu(test_data,mr_pairs,test_out)
    write_betweengroup_edit(test_data, mr_pairs, test_out)
    write_betweengroup_slot(test_data, mr_pairs, test_out)


def evaluate_ner_model(tagged_file):
    scores = []
    mr_tag_dict = read_into_groups(tagged_file)
    for mr in mr_tag_dict:
        gt_slot = []
        for slot in mr.split(","):
            if not slot.strip().startswith("family"):
                gt_slot.append(slot)
        gt = ",".join(gt_slot)
        for tag in mr_tag_dict[mr]:
            pred_slot = []
            for slot in tag.split(","):
                if not (slot.strip().startswith("family") or slot.strip().startswith("negate")):
                    pred_slot.append(slot)
            pred = ",".join(pred_slot)
            score = evaluate_slot_sim(gt,pred)
            scores.append(score)
    acc = np.mean(np.array(scores))
    print(acc)

def calculate_ground_truth(f_in):
    mr_pairs = read_mr_pair_file(f_in)
    evaluate_gold(mr_pairs)


def process_table():
    str = """bleu                    & 0.298979286   & 0.161825372   & 0.87904367    & 0.73570803   \\ \hline
dan                     & 0.265240195   & 0.231764987   & 0.80979497    & 0.7616405    \\ \hline
edit                    & 0.053573342   & 0.17074284    & 0.55327899    & 0.39489102   \\ \hline
slot                    & 0.219619997   & 0.237006949   & 0.77610745    & 0.63530431   \\ \hline
one-hot                 & -0.12276416   & -0.10449101   & 0.6478378     & 0.58166724   \\ \hline
sentBERT                & 0.707444459   & 0.673425218   & 0.30381901    & 0.46933566   \\ \hline
word2vec                & 0.290532627   & 0.265885046   & 0.67135597    & 0.47939191   \\ \hline
w2vsif                  & -0.20296333   & -0.27850558   & 0.57984284    & 0.74620092   \\ \hline"""
    nt = []
    for c in str.split():
        if "0" in c:
            num = "{:.4f}".format(float(c))
            nt.append(num.strip('0'))
        elif c == "\\":
            nt.append("\\\\")
        elif c == "\\hline":
            nt.append("\hline\n")
        else:
            nt.append(c)
    print(" ".join(nt))

if __name__ == '__main__':
    data_dir = "/Users/yfeng/Public/Study/20Spring/11727/project/e2e-cleaning/cleaned-data/"
    out_dir = "./test_out/"
    mr_pair_file = "./test_out/between-group/between_group_mr.txt"
    test_csv = data_dir+"test-fixed"
    dev_csv = data_dir+"devel-fixed.no-ol"
    test_out = out_dir + "test-"
    dev_out = out_dir + "dev-"
    negation_csv = "negation_set.csv"
    corr_csv = "corr.csv"
    # calculate_ground_truth(mr_pair_file)
    process_table()
    # run_subset_ingroup(test_csv,test_out)
    # run_subset_ingroup(dev_csv,dev_out)
    # run_subset_between_group(test_csv, test_out,mr_pair_file)
    # extract_differ_by_one_pairs(test_out)
    # read_and_eval_negation(negation_csv,test_out+"negation-")
    # read_and_eval_corr(corr_csv, test_out+"corr-")
    # evaluate_ner_model("/Users/yfeng/Public/Study/20Spring/11727/project/measure_similarity/test_out/test-nl-by-group.tagged.txt")