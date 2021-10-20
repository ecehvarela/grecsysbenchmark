'''


'''
import GRS_Evaluation.ratings_agg.Parameters as p
from GRS_Evaluation.ratings_agg.Model import *
#from GRS_Evaluation.ratings_agg.GRS_ratings_agg import *
from GRS_Evaluation.ratings_agg.GRS_ratings_agg import *

from datetime import datetime, date


def prepare_parameters(gs, ng, gt, agg, af, rs, m,  data, dpath, mpath):
    # check for Group size
    if gs == "all":
        gs = p.group_size
    else:
        p.group_size = [gs]

    # check for Group type
    if gt == "all":
        gt = p.group_type
    else:
        p.group_type = [gt]

    if af == "all":
        af = p.agg_functions
    else:
        p.agg_functions = [af]

    if rs == "all":
        rs = p.rec_system
    else:
        p.rec_system = [rs]

    if m == "all":
        m = p.metrics
    else:
        p.metrics = [m]

    if data == "all":
        data = p.datasets
    else:
        p.datasets = [data]

    if dpath.endswith("/"):
        dpath = dpath[:-1]
        p.data_path = dpath

    if mpath.endswith("/"):
        mpath = mpath[:-1]
        p.models_path = mpath


def write_test_separator():

    now = datetime.now()
    dt_string = now.strftime("%m-%d-%Y-%H:%M:%S")
    delimiters = "="*20
    sepatator_text = "SEPARATOR"
    text = "{} {}_{} {}".format(delimiters, dt_string, sepatator_text,delimiters)
    print(text)

    filename = p.logs_path + "/" + text + ".txt"
    with open(filename , "w") as f:
        f.write("")


def write_test_results(data, model, group_type, group_size, function, metric_for_result, grs, ng):
    print("\t\t\t\t\t\t Save results as {}".format(
        data + "_" + model + "_" + group_type + "_" + group_size +
        "_" + function + "_" + metric_for_result + "_metrics.csv"))

    now = datetime.now()
    dt_string = now.strftime("%m-%d-%Y-%H:%M:%S")
    delimiters = "=" * 20
    sepatator_text = data + "_" + model + "_" + group_type + "_" + group_size + "_" + function \
                     + "_" + metric_for_result + "_metrics_" + str(ng)
    text = "{}_{}".format(sepatator_text, dt_string)

    filename = p.logs_path + "/" + text + ".csv"

    header = ",".join([metric for metric in p.metrics]) + "\n"

    #for metric in p.metrics:
    #    print("{}: {}".format(metric, grs.eval.get_avg_metric_result(metric)))

    results_text = ",".join([str(grs.eval.get_avg_metric_result(metric)) for metric in p.metrics]) + "\n"

    # Write the header
    with open(filename , "w") as f:
        f.write(header)

    # Write the results
    with open(filename , "a") as f:
        f.write(results_text)


def write_metric_results_df(df, agg, data, model, group_type, group_size, function, metric, grs):
    #columns = ["agg_method", "dataset", "rs", "g_type", "g_size", "agg_func", "metric", "results"]
    #print("#" * 80)
    #print("#" * 80)
    #print("Model: {}, Function: {}, Metric: {}".format(model, function, metric))
    #print(grs.eval.get_metric_results(metric))
    results = grs.eval.get_metric_results(metric)[0]

    df = df.append({ "agg_method" : agg,
                     "dataset" : data,
                     "rs" : model,
                     "g_type" : group_type,
                     "g_size" : group_size,
                     "agg_func" : function,
                     "metric" : metric,
                     "results" : results
                     }, ignore_index=True)

    return df
    #print("#" * 80)
    #print("#" * 80)


def form_data_filename(data):
    '''

    '''
    data_filename = p.data_path+"/"+data.upper()+"/ratings.csv"
    return data_filename


def form_matrices_filename(data):
    '''

    '''
    matrices_filename = p.data_path+"/"+data.upper()+"/matrices.obj"
    return matrices_filename

def form_model_filename(data, model):
    '''

    '''
    model_filename = p.models_path + "/" + data + "_" + model + ".model"
    return model_filename

def save_results_df(dataset, df):
    #print(df)
    now = datetime.now()
    dt_string = now.strftime("%m_%d_%Y_%H_%M_%S")

    filename = p.logs_path + "/" + dataset + "_results" + "_" + dt_string + ".df"
    #filename = p.logs_path + "/" + dataset +  "_results.df"
    df.to_pickle(filename)
    print(filename, " saved.")


def run(gs, ng, gt, agg, af, rs, m, data, dpath, mpath):
    '''

    '''
    metric_for_result = m
    prepare_parameters(gs, ng, gt, agg, af, rs, m, data, dpath, mpath)
    write_test_separator()

    # Create an empty df to store the list ofresults
    columns = ["agg_method", "dataset", "rs", "g_type", "g_size", "agg_func", "metric", "results"]
    results_df = pd.DataFrame(columns=columns)

    print("== Evaluations ==")
    print("Group size:", p.group_size)
    print("Number of groups:", ng)
    print("Group type:", p.group_type)
    print("Agg strategy:", agg)
    print("Agg function:", p.agg_functions)
    print("Rec System:", p.rec_system)
    print("Metrics:", p.metrics)
    print("Dataset:", p.datasets)
    print("Data path:", p.data_path)
    print("Model path:", p.models_path)

    print("@k:", p.at_k)
    print("*" * 50)
    for data in p.datasets:
        print("Using dataset: {}".format(data))

        for model in p.rec_system:
            print("\t Using RS model: {}".format(model))
            grs = GRS(model, data)
            print(grs.get_model_name())

            print("\t ~~~ load the model and dataset ~~~")

            ratings_filename = form_data_filename(data)
            print("Data filename:", ratings_filename)

            model_filename = form_model_filename(data, model)
            print("Model filename:", model_filename)
            grs.load_data_and_model(ratings_filename, model_filename)

            matrices_fileame = form_matrices_filename(data) #this is for CB
            print("Matrices filename:", matrices_fileame)
            grs.load_matrices(matrices_fileame)

            for group_type in p.group_type:
                print("\t\t Using Group type: {}".format(group_type))

                for group_size in p.group_size:
                    grs.reset_values()

                    print("\t\t\t Using Group size: {}".format(group_size))
                    print("\t\t\t Generate the {} ({} and {}) groups".format(ng, group_type, group_size))
                    grs.form_groups(ng, group_size, group_type)
                    #grs.view_groups()
                    grs.get_G_common_items_not_rated()
                    #grs.recommend_for_group_members()
                    grs.get_members_predictions_df()

                    for function in p.agg_functions: # here I work for all formed groups.
                        print("")
                        print("*" * 50)
                        print("\t\t\t\t Using function: {}".format(function))
                        print("\t\t\t\t For each of the {} groups".format(ng))
                        print("\t\t\t\t Generate each member's recommendation")
                        print("\t\t\t\t Generate DF with members' predictions")
                        print("\t\t\t\t Apply the Aggregation function {}".format(function))
                        print("\t\t\t\t Generate Group recommendation")
                        grs.group_recommendations(function)

                        for metric in p.metrics:
                            print("\t\t\t\t\t\t Calculate {} for all groups ".format(metric))
                            grs.eval.get_evaluations(metric, grs._G_recommendations_by_func[function],
                                                     grs._G_member_predictions_dfs, grs.cosine_sim_matrix_df,
                                                     model, grs._iids) #, grs.G_member_predictions_list)

                            print("\t\t\t\t\t\t Average {} results".format(metric))

                            print("\t\t\t\t\t\t {}".format("-"*50))

                            results_df = write_metric_results_df(results_df, agg, data, model, group_type, group_size,
                                                                 function, metric, grs)

                        write_test_results(data, model, group_type, group_size, function, metric_for_result, grs, ng)



        save_results_df(data, results_df)



###########################################

if __name__ == "__main__":
    run()