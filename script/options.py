import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mr1", type=str, default="name[Alimentum], area[city centre], familyFriendly[no]", help="meaning representation 1 ")
    parser.add_argument("--mr2", type=str, default="", help="meaning representation 2")
    parser.add_argument("--input_file", type=str, default='cleaned_devset.csv', help="input file")
    parser.add_argument("--in_group_sim", action="store_true", help="get in group similarity")
    parser.add_argument("--get_sents", action="store_true", help="whether to get all the sentences for meaning representation 1")
    parser.add_argument("--bert_vec", action="store_true", help="use bert sentence vector")
    parser.add_argument("--roberta_vec", action="store_true", help="use roberta sentence vector")


    # sanity check
    parser.add_argument("--sanity_check", action='store_true',
                        help="Use small data set to run for sanity check")
    # reproducibility
    parser.add_argument("--random_seed", type=int, default=9,
                        help="Random seed (>0, set a specific seed).")
    # Mode
    parser.add_argument("--cross_validate", action='store_true',
                        help="Use cross-validation on the whole data (including 2018 train and test)")


    return parser.parse_args()