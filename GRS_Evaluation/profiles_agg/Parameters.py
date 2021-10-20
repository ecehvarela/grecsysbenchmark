'''

'''

NUM_GROUPS = 2
NUM_MEMBERS = 4
NUM_ITEMS_TO_RECOMMEND = 100

# Group sizes
size = {"S":(2,6),
        "M":(7,20),
        "L":(21,50),
        "VL":(51, 100)}

group_size = ["s","m","l","vl"]
group_type = ["r","s","d","rl"]
agg_functions = ["avg","add","app","awm","bc","lm","mp","mrp","mul","pop"]
rec_system = ["ibcf", "ubcf", "iucf", "cb", "hybrid", "svd", "ncf"]
metrics = ["hr","ndcg","diversity","coverage"]
datasets = ["thi", "op", "plg", "bea", "ggf", "dm", "camra2011"]

data_path = "../data"
models_path = "../GRS_Evaluation/ratings_agg/models"

logs_path = "../GRS_Evaluation/profiles_agg/logs"
#profile_logs_path = "../GRS_Evaluation/profiles_agg/logs"

sim_options = {'name': 'cosine'}

alpha = 0.6
at_k = 20 #5
min_pop = 20 #50
top_bc = 10