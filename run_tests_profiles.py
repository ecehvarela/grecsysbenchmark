'''


'''
import GRS_Evaluation.profiles_agg.Parameters as p
from GRS_Evaluation.profiles_agg.Model import *
from GRS_Evaluation.profiles_agg.GRS_ratings_agg import *

from datetime import datetime, date
import os
import pickle


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
    print("\t\t\t\t\t\t Save results as {}".format(p.logs_path + "/" +
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

def form_metadata_filename(data):
    '''

    '''
    path = p.data_path + "/" + data.upper()
    print("dp:",path)
    files = os.listdir(path)
    #os.chdir(os.pardir)
    #currentDirectory = os.getcwd()
    #print("Current working directory:", currentDirectory)

    #print(files)
    #meta_file = [currentDirectory+'/data/'+ data.upper() + "/" + file for file in files if "meta" in file][0]
    meta_file = [path + "/" + file for file in files if "meta" in file][0]
    print(meta_file)

    return meta_file


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

    filename = p.logs_path + "/" + dataset +"_results" + "_" + dt_string + ".df"
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

            ###matrices_fileame = form_matrices_filename(data) #this is for CB
            ###print("Matrices filename:", matrices_fileame)

            """ Removed on 08/29/2021 for this test with CAMRA2011
            meta_in_filename = form_metadata_filename(data)
            print("Meta filename:",meta_in_filename)
            meta_out_filename = form_matrices_filename(data)
            print("Meta OUT filename", meta_out_filename)
            grs.get_matrices(meta_in_filename, meta_out_filename, save=0)
            """

            for group_type in p.group_type:
                print("\t\t Using Group type: {}".format(group_type))

                for group_size in p.group_size:
                    grs.reset_values()

                    print("\t\t\t Using Group size: {}".format(group_size))
                    print("\t\t\t Generate the {} ({} and {}) groups".format(ng, group_type, group_size))
                    grs.form_groups(ng, group_size, group_type)
                    grs.view_groups()
                    ###grs.get_G_common_items_not_rated()
                    #grs.recommend_for_group_members()
                    ###grs.get_members_predictions_df()

                    for function in p.agg_functions: # here I work for all formed groups.
                        #grs.delete_model()

                        print("")
                        print("*" * 50)
                        print("\t\t\t\t Using function: {}".format(function))
                        print("\t\t\t\t For each of the {} groups".format(ng))
                        print("\t\t\t\t\t Generate the 'VIRTUAL USER'")
                        print("\t\t\t\t\t Add the 'VIRTUAL USER' ratings to the ratings dataset")

                        #grs.generate_virtual_user_pivot_dataset()
                        grs.generate_virtual_user_profile(function)
                        ##grs.generate_virtual_user_pivot_dataset()
                        grs.get_all_user_items_not_rated()

                        print("\t\t\t\t With the final dataset, train the model")
                        grs.create_model()

                        print("\t\t\t\t Generate DF with VIRTUAL USERS' predictions")
                        print("\t\t\t\t Get the VIRTUAL USERS missing ratings")
                        print("\t\t\t\t Generate VIRTUAL USERS recommendations")
                        ###grs.group_recommendations(function)
                        #sys.exit(8)
                        grs.get_all_user_predictions_df()
                        grs.get_all_user_recommendations()
                        #grs.testing()

                        for metric in p.metrics:
                            print("\t\t\t\t\t\t Calculate {} for all groups ".format(metric))
                            grs.eval.get_evaluations(metric, grs._user_predicted_df, grs.cosine_sim_matrix_df,
                                                     grs._user_recommendations, model,
                                                     grs.get_groups(), grs.get_virtual_users(), grs._iids)
                            #grs.eval.get_evaluations(metric, grs._G_recommendations_by_func[function],
                            #                         grs._G_member_predictions_dfs, grs.cosine_sim_matrix_df,
                            #                         model) #, grs.G_member_predictions_list)

                            print("\t\t\t\t\t\t Average {} results".format(metric))


                            #print("\t\t\t\t\t\t Save results as {}".format(data+"_"+model+"_"+group_type+"_"+group_size+
                            #                                             "_"+function+"_"+metric+".csv"))

                            print("\t\t\t\t\t\t {}".format("-"*50))

                            results_df = write_metric_results_df(results_df, agg, data, model, group_type, group_size, function, metric, grs)

                        write_test_results(data, model, group_type, group_size, function, metric_for_result, grs, ng)

                        grs.delete_model()
            del grs


        save_results_df(data, results_df)
        #print(results_df)




###########################################

if __name__ == "__main__":
    run()