from itertools import combinations
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import matplotlib.pyplot as plt
import re
from ast import literal_eval

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
    # return score
    print(score)

def evaluate_in_group(sents,mode):
    score = 0
    total = 0
    if mode == "slot":
        x = [i for i in range(len(sents))]
        pairs = list(combinations(x, 2))
        for p in pairs:
            score+=evaluate_slot_sim(sents[p[0]],sents[p[1]])
            total+=1
    elif mode == "bleu":
        for i in sents:
            ref = sents[:]
            ref.remove(i)
            score += evaluate_bleu_sim(i, ref)
            total += 1
    score/=total
    print("in group similarirty:{:.2f}%".format(score*100))


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

def evaluate_bleu(all_groups):
    evaluate_in_group(all_groups[0],"bleu")
    evaluate_between_group(all_groups[0], all_groups[1],"bleu")
    evaluate_between_group(all_groups[0], all_groups[2],"bleu")


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

def plot_heat_map():
    corr_txt = """[[1.         0.77076674 0.8851582  0.82740533 0.6088928  0.6512091 ]
     [0.77076674 0.99999976 0.74573034 0.73024035 0.59289664 0.64044225]
     [0.8851582  0.74573034 0.9999998  0.8800323  0.7052758  0.76022315]
     [0.82740533 0.73024035 0.8800323  1.0000002  0.6495909  0.7044904 ]
     [0.6088928  0.59289664 0.7052758  0.6495909  0.99999994 0.9122772 ]
     [0.6512091  0.64044225 0.76022315 0.7044904  0.9122772  1.0000002 ]]"""
    """[[1.         0.9100553  0.8302746  0.9322071  0.9696125  0.8144195
  0.7220932  0.7669071  0.7716822  0.76309144 0.7891823  0.75822264]
 [0.9100553  0.99999964 0.83549595 0.85186857 0.8728513  0.80914104
  0.68129945 0.72872657 0.7444346  0.72839046 0.7510123  0.7073161 ]
 [0.8302746  0.83549595 0.99999994 0.788553   0.80538857 0.9639911
  0.75047594 0.8003451  0.77649575 0.7517127  0.80455494 0.72671187]
 [0.9322071  0.85186857 0.788553   1.0000001  0.94790137 0.76269567
  0.68369615 0.68600374 0.7062142  0.69739825 0.7070118  0.69376403]
 [0.9696125  0.8728513  0.80538857 0.94790137 0.99999994 0.79174095
  0.6992121  0.7274179  0.71888906 0.72485113 0.73333025 0.71898305]
 [0.8144195  0.80914104 0.9639911  0.76269567 0.79174095 0.9999999
  0.73631823 0.77739406 0.75142336 0.72660714 0.7784214  0.69413704]
 [0.7220932  0.68129945 0.75047594 0.68369615 0.6992121  0.73631823
  0.99999976 0.912277   0.87506235 0.8933116  0.8876017  0.8837184 ]
 [0.7669071  0.72872657 0.8003451  0.68600374 0.7274179  0.77739406
  0.912277   0.9999999  0.95433575 0.9425238  0.956671   0.9037704 ]
 [0.7716822  0.7444346  0.77649575 0.7062142  0.71888906 0.75142336
  0.87506235 0.95433575 0.9999999  0.94145346 0.9692459  0.87572694]
 [0.76309144 0.72839046 0.7517127  0.69739825 0.72485113 0.72660714
  0.8933116  0.9425238  0.94145346 0.9999999  0.93585306 0.87065315]
 [0.7891823  0.7510123  0.80455494 0.7070118  0.73333025 0.7784214
  0.8876017  0.956671   0.9692459  0.93585306 1.0000002  0.89542156]
 [0.75822264 0.7073161  0.72671187 0.69376403 0.71898305 0.69413704
  0.8837184  0.9037704  0.87572694 0.87065315 0.89542156 0.9999998 ]]"""
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

file_in = "/Users/yfeng/Public/Study/20Spring/11727/project/neural-template-gen/e2e_example/presentation.tagged.txt"
all_groups = read_into_groups(file_in)
plot_heat_map()
# evaluate_slot(all_groups)
# evaluate_bleu(all_groups)
# evaluate_gold()


