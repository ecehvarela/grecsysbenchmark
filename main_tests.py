'''

-gs s -ng 10 -gt r -agg r -af all -rs all -m all -d baby
'''
import argparse
#import my_tests.run_tests as tests
import run_tests as tests
import my_tests.run_tests_profiles as tests_p
import sys

def display_arguments(*argv):
    print("Group size:", argv[0])
    print("Groups:", argv[1])
    print("Group type:", argv[2])
    print("Aggregation strategy:", argv[3])
    print("Aggregation function:", argv[4])
    print("Recommender System:", argv[5])
    print("Metrics:", argv[6])
    print("Dataset:", argv[7])
    print("Data path:", argv[8])
    print("Model path:", argv[9])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main file for survey")
    # group size
    parser.add_argument("-gs", default="s", type=str, choices=["s","m","l","vl","all"], help="Group size [S]:small, [M]:medium, [L]:large, [VL]:very large, [ALL]: all sizes")
    # number of groups
    parser.add_argument("-ng", default=10, type=int, help="Number of groups")
    # group type
    parser.add_argument("-gt", default="r", type=str, choices=["r","s","d","rl","all"], help="Group type [R]:random, [S]:similar, [D]:dissimilar, [RL]:realistic, [All]: all sizes")
    # aggregation strategy
    parser.add_argument("-agg", default="r", type=str, choices=["r","p"], help="Aggregation strategy [R]:ratings, [P]:preferences")
    # aggregation function
    parser.add_argument("-af", default="awm", type=str,
                        choices=["avg","add","app","awm","bc","lm","mp","mrp","mul","pop","all"],
                        help="Aggregation function "
                             "[AVG]: average, "
                             "[ADD]: additive, "
                             "[APP]: approval, "
                             "[AWM]: average without misery, "
                             "[BC]: borda count,"
                             "[LM]: least misery, "
                             "[MP]: most pleasure,"
                             "[MRP]: most respected person, "
                             "[MUL]: multiplicative,"
                             "[POP]: popularity,"
                             "[All]: all"
                        )

    # RS model
    parser.add_argument("-rs", default="all", type=str,
                        choices=["ibcf", "ubcf", "iucf", "cb", "hybrid", "svd", "ncf","all"],
                        help="Recommender system model "
                             "[IBCF]: Item-based CF, "
                             "[UBCF]: User-based CF, "
                             "[IUCF]: IICF+UUCF, "
                             "[CB]: Content Based, "
                             "[Hybrid]: hybrid,"
                             "[SVD]: SVD++,"
                             "[NCF]: Neural CF,"
                             "[All]: all")

    # Metrics
    parser.add_argument("-m", default="all", type=str,
                        choices=["hr","ndcg","diversity","coverage","all"],
                        help="Evaluation metrics "
                             "[HR]: hit rate,"
                             "[nDCG]: nDCG,"
                             "[Diversity]: Diversity,"
                             "[Coverage]: Coverage,"
                             "[All]: all")


    # dataset
    parser.add_argument("-d", type=str,
                        choices=["dm", "ggf", "music", "plg", "op", "thi", "ml", "all", "camra2011"],
                        required=True,
                        help="Datasets "
                             "DM, "
                             "GGF, "
                             "MUSIC,"
                             "PLG,"
                             "OP,"
                             "THI,"
                             "ML,"
                             "All,"
                             "CAMRA20211")

    # data path
    parser.add_argument("-dpath", type=str,
                        required=True,
                        help="Path for the data folder (i.e., ../data)")

    # model path
    parser.add_argument("-mpath", type=str,
                        required=True,
                        help="Path for the model folder (i.e, ../GRS_Evaluation/ratings_agg/models")

    args = parser.parse_args()

    gs = args.gs.lower()
    ng = args.ng
    if ng < 1:
        print("[ERROR] At least it should be 1 group!")
        exit(1)
    gt = args.gt.lower()
    agg = args.agg.lower()
    af = args.af.lower()
    rs = args.rs.lower()
    m = args.m.lower()
    data = args.d.lower()
    dpath = args.dpath.lower()
    mpath = args.mpath.lower()

    display_arguments(gs, ng, gt, agg, af, rs, m, data, dpath, mpath)
    if agg == 'r':
        tests.run(gs, ng, gt, agg, af, rs, m, data, dpath, mpath)
    else:
        tests_p.run(gs, ng, gt, agg, af, rs, m, data, dpath, mpath)
