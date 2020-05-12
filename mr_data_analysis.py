from string_level import *


def compare_two_buckets(six_slots,eight_slots):
    num_diff_slot_value = defaultdict(list)
    for l1 in range(len(six_slots)):
        for l2 in range(l1, len(eight_slots)):
            # [mr, slots, values]
            slot_diff = set(six_slots[l1][1]).difference(eight_slots[l2][1])
            value_diff = set(six_slots[l1][2]).difference(eight_slots[l2][2])
            num_diff_slot_value["{}_{}".format(len(slot_diff), len(value_diff))].append(
                ["differ by slot {}, differ by value {}".format(" ".join(slot_diff), " ".join(value_diff)),
                 six_slots[l1][0],
                 eight_slots[l2][0]])
    print("done")

def bucket_by_num_diff(list_with_same_num_slots):
    num_diff_slot_value = defaultdict(list)
    for l1 in range(len(list_with_same_num_slots)):
        for l2 in range(l1,len(list_with_same_num_slots)):
            # [mr, slots, values]
            slot_diff = set(list_with_same_num_slots[l1][1]).difference(list_with_same_num_slots[l2][1])
            value_diff = set(list_with_same_num_slots[l1][2]).difference(list_with_same_num_slots[l2][2])
            num_diff_slot_value["{}_{}".format(len(slot_diff),len(value_diff))].append(["differ by slot {}, differ by value {}".format(" ".join(slot_diff)," ".join(value_diff)),
                                                                                        list_with_same_num_slots[l1][0],
                                                                                        list_with_same_num_slots[l2][0]])
    print("done")

def analyze_eight_slots(cleaned_mr_pairs):
    key_with_most_items = "area_customer rating_eatType_familyFriendly_food_name_near_priceRange"  # 261
    differ_by_bool = []
    differ_by_other = []
    num_diff = defaultdict(list)
    for mr1 in cleaned_mr_pairs[key_with_most_items]:
        for mr2 in cleaned_mr_pairs[key_with_most_items]:
            diff = set(mr1[1]).difference(mr2[1])
            if len(diff) == 1:
                diff_value = diff.pop()
                diff_slot = key_with_most_items.split("_")[mr1[1].index(diff_value)]
                if diff_slot == "familyFriendly":
                    differ_by_bool.append((mr1[0], mr2[0]))
                elif diff_slot in LABEL:
                    differ_by_other.append((mr1[0], mr2[0]))
            else:
                num_diff[len(diff)].append([mr1[0], mr2[0]])


def extract_differ_by_one_pairs(test_csv,test_out):
    LABEL = ["name", "near","area","food"]
    test_data = read_e2e_csv(test_csv)
    differ_by_bool = []
    differ_by_other = []
    all_mr_slots = defaultdict(list)
    num_diff = defaultdict(list)
    six_slots = []
    key_with_most_items = "area_customer rating_eatType_familyFriendly_food_name_near_priceRange"  # 261
    eight_slots = []
    for k in test_data:
        mr_pair = extract_slots_pair(k)
        sorted_mr_pair = sorted(mr_pair,key=lambda x: x[0])
        mr_slots = [m[0] for m in sorted_mr_pair]
        # we only care about family-friendly and customer-rating different
        if "familyFriendly" in mr_slots and "customer rating" in mr_slots:
            all_mr_slots["_".join(mr_slots)].append((k,[m[1] for m in sorted_mr_pair])) #{slots:[(mr,value)]}
    cleaned_mr_pairs = {k: v for k, v in all_mr_slots.items() if len(v) > 1}
    for k in all_mr_slots:
        if len(k.split("_")) == 6:
            for i in range(len(all_mr_slots[k])):
                six_slots.append([all_mr_slots[k][i][0],k.split("_"),all_mr_slots[k][i][1]])
    for i in range(len(all_mr_slots[key_with_most_items])):
        eight_slots.append([all_mr_slots[key_with_most_items][i][0],key_with_most_items.split("_"), all_mr_slots[key_with_most_items][i][1]])
    # bucket_by_num_diff(six_slots)
    compare_two_buckets(six_slots, eight_slots)


if __name__ == '__main__':
    out_dir = "./test_out/"
    mr_pair_file = "./test_out/between-group/between_group_mr.txt"
    test_csv = "cleaned_testset"
    dev_csv = "cleaned_devset"
    test_out = out_dir + "test-"
    dev_out = out_dir + "dev-"
    extract_differ_by_one_pairs(test_csv,test_out)