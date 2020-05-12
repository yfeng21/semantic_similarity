import csv

e2e_keys = ["name", "eatType", "food", "priceRange", "customerrating", "area", "near"]
#data_dir = "/Users/yfeng/Public/Study/20Spring/11727/project/neural-template-gen/e2e_example"
data_dir = "/home/yulan/direction_nlg/template_generation/neural-template-gen/data/e2e_aligned/"
data_set = "train"


with open(data_dir + data_set+".tsv", 'wt') as out_file:
    with open(data_dir + data_set+".txt", "r") as f:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for line in f.readlines():
            contents, tags = [l.split() for l in line.split("<eos>|||")]
            token_tag = [[tok,"0"] for tok in contents]
            for t in tags:
                tag_tuple = [int(ti) for ti in t.split(",")]
                if tag_tuple[2] < len(e2e_keys):
                    if tag_tuple[1] - tag_tuple[0] == 1:
                        token_tag[tag_tuple[0]] = [contents[tag_tuple[0]],e2e_keys[tag_tuple[2]]]
                    else:
                        token_tag[tag_tuple[0]] = [" ".join(contents[tag_tuple[0]:tag_tuple[1]]), e2e_keys[tag_tuple[2]]]
                        for e in range(tag_tuple[0]+1,tag_tuple[1]):
                            token_tag[e] = []
            if contents[-1] != ".":
                token_tag.append([".","0"])
            for row in token_tag:
                if len(row) > 0:
                    tsv_writer.writerow(row)




