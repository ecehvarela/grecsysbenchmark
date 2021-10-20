import GRS_Evaluation.ratings_agg.Model as m
import GRS_Evaluation.ratings_agg.Parameters as p
import GRS_Evaluation.ratings_agg.AggFunctions as af
import GRS_Evaluation.Evaluator as eval
import GRS_Evaluation.sim_disim_groups as sd

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import random
import sys

class GRS():

    def __init__(self, model, dataset):
        self._model = model
        self._name = model
        self._dataset_name = dataset
        self._agg_strategy = 'ratings'

        self.reset_values()

        self.agg = af.AggFunctions()
        self.eval = eval.Evaluator()

    def reset_values(self):
        self._group_type = None
        self._group_size = None
        self._agg_function = None
        self._groups = []
        self._func_recommendations = defaultdict(list)
        self._G_func_recommendations = []
        self._group_unrated_items = []
        self._G_member_predictions_dfs = []
        self._G_members_has_rated_number_per_group = []

        self._G_recommendations_by_func = defaultdict(list)

    '''
    def __init__(self, algos, functions, item_category):
        
        self._name = algos
        print("Group Recommender System using Ratings Aggregation for {}".format(self._name))
        self._algo = None
        self._dataset = None
        self._rs = None
        self._groups = []
        self._agg_strategy = "ratings"
        self._functions = functions
        self._dataset_name = item_category
        self._func_recommendations = defaultdict(list)
        #self._metric = metric
        self.agg = af.AggFunctions()
        self.eval = eval.Evaluator()

    '''

    def test_matrices(self):
        print("Cosine similarity")
        print(self.cosine_sim_matrix_df.head())
        print()
        print("tfidf matrix")
        print(self.tfid_matrix_df.head())

    def get_model_name(self):
        return self._model


    def set_metric(self, metric):
        '''

        '''
        self._metric = metric


    def load_matrices(self, matrices_filename):
        '''

        '''

        print("Loading matrices in: {}".format(matrices_filename))
        data = m.Model(self._name).load_matrices_data(matrices_filename)
        self.tfid_matrix_df = data[0]
        self.cosine_sim_matrix_df = data[1]


    def load_data_and_model(self, ratings_filename, model_filename):
        '''

        '''
        # load the dataset
        rs = m.Model(self._name)
        rs.load_data(ratings_filename)
        self._dataset = rs._data
        #print(self._dataset)

        if "IUCF" in model_filename.upper():
            print("This is the combination of IBCF and UBCF")

            print("First load the UBCF model")
            model_filename = model_filename.replace("iucf","ubcf")
            print("Loading {}".format(model_filename))
            self._rs = rs.load_model(model_filename)

            print("Second, load the IBCF model")
            model_filename = model_filename.replace("ubcf", "ibcf")
            print("Loading {}".format(model_filename))
            self._rs2 = rs.load_model(model_filename)

        elif "HYBRID" in model_filename.upper():
            print("This is the hybrid model, combining SVD and CB")

            print("First, load the SVD model")
            model_filename = model_filename.replace("hybrid", "svd")
            print("Loading {}".format(model_filename))
            self._rs = rs.load_model(model_filename)

            print("Second, load the CB model")
            model_filename = model_filename.replace("svd", "cb")
            print("Loading {}".format(model_filename))
            self._rs2 = rs.load_model(model_filename)


        #elif "CB" == self._name.upper():
        #    print("Model is {}".format(self._name))
        #    print()
        #    print(model_filename)
        #    sys.exit(7)

        else:
            # load the trained rs model
            self._rs = rs.load_model(model_filename)
            #print(self._rs)

        # Get the list of uids
        self._uids = self._dataset['uid'].unique()
        print("There are {} users".format(len(self._uids)))
        print("The mininum users id:", self._uids.min())
        print("The maximum users id:", self._uids.max())
        self._iids = self._dataset['iid'].unique()
        print("Total items:", len(self._iids))
        print("The mininum items id:", self._iids.min())
        print("The maximum items id:", self._iids.max())


    def get_random_groups_for_size(self, num_groups, size='s'):
        '''

        '''
        group_sizes = []
        s_init, s_end = p.size[size.upper()]
        print("Creating {} groups of size between {} and {}".format(num_groups, s_init, s_end))
        for x in range(num_groups):
            n = random.randrange(s_init, s_end+1)
            print("Group of {}".format(n))
            group_sizes.append(n)

        return group_sizes


    def form_groups(self, num_groups, size, type='r'):
        '''
            Create a list of groups depending on their type
        '''
        print("Form {} groups of size {}".format(num_groups, size.upper()))

        group_member_path =  "../data/CAMRA2011/groupMember.txt"

        group_members = dict()
        with open(group_member_path, "r") as f:
            f.readlines

            for line in f:
                group, members = line.split()
                members = list(map(int, members.split(",")))
                group_members[int(group)] = members

        print(f"There are {len(group_members)} groups")
        group_ids = list(group_members.keys())
        group_ids.sort()


        self.group_type = type
        self.group_size = size


        if (self._dataset_name == "camra2011"):
            print("Just for CAMRa2011")

            selected_groups = np.random.choice(group_ids, num_groups, replace=False)
            print("Selected groups:", selected_groups)
            for g in selected_groups:
                G = group_members.get(g)
                self._groups.append(G)
            print(self._groups)


        # group_sizes = self.get_random_groups_for_size(num_groups, size)
        # if type == 'r':
        #     print("Insider 'r'")
        #     for g in group_sizes:
        #         G = np.random.choice(self._uids, g, replace=False)
        #         self._groups.append(G)
        #
        #     print(self._groups)







    def get_groups(self):
        '''
            Return the list of groups
        '''
        return self._groups


    def view_groups(self):
        '''
            Display the groups and their members
        '''
        for i, G in enumerate(self._groups):
            print("Group", i+1 ,":", ",".join([str(e) for e in G.tolist()]))

    def items_rated_for_group_member(self, member):
        print("Getting iid list for member:", member)
        iids_rated = self._dataset.loc[self._dataset['uid'] == member, 'iid']
        ##print("\t items rated by user {}: {}".format(member, len(iids_rated)))
        return iids_rated



    def get_G_common_items_not_rated(self):

        # For each Group
        for i, G in enumerate(self._groups):
            print("=== Group {} ===".format(i+1))
            self.G_member_has_rated = {}
            self.G_members_has_rated_number = {}
            self.G_member_not_rated = {}

            # Get a list of iids that each group member has rated
            for member in G:
                #print("Getting iid list for member:", member)
                #iids_rated = self._dataset.loc[self._dataset['uid'] == member, 'iid']
                ##print("\t items rated by user {}: {}".format(member, len(iids_rated)))
                iids_rated = self.items_rated_for_group_member(member)
                # Save list into a dictionary
                self.G_member_has_rated[member] = iids_rated
                self.G_members_has_rated_number[member] = len(iids_rated)

            # Get the list of iids that each group members has not rated
            #for member in G:
                #iids_rated = self.G_member_has_rated[member]
                iids_to_pred = np.setdiff1d(self._iids, iids_rated)
                ##print("user {} has not rated: {}".format(member, len(iids_to_pred)))
                ##print("\t Some of these items are:", iids_to_pred)
                # Save list into a dictionary
                self.G_member_not_rated[member] = iids_to_pred

            # Get the list of common no rated items for the group
                set_list = []
            # Get all the sets in a single list
            #for member in G:
                iids_to_pred = self.G_member_not_rated[member]
                set_list.append(set(iids_to_pred))

            # Get the intersection of all sets within the set_list
            #self.iids_G_common_not_rated = set.intersection(*set_list)
            iids_G_common_not_rated = set.intersection(*set_list)
            print("The group members have {} items not rated in common".format(len(iids_G_common_not_rated)))
            self._group_unrated_items.append(iids_G_common_not_rated)

            self._G_members_has_rated_number_per_group.append(self.G_members_has_rated_number)

        print(self._G_members_has_rated_number_per_group)
            #-------------------------------------------

    def get_iucf_members_predictions_df(self):
        # For each group member predict the rating for each movie that the Group hasn't rated
        self.G_member_predictions = {}
        for i, G in enumerate(self._groups):
            print("=== Predictions for group {} ===".format(i+1))
            print("To predict:",len(self._group_unrated_items[i]))
            for member in G:
                testset = [[member, iid, 3.0] for iid in self._group_unrated_items[i]]
                # testset = [[member, iid, 3.0] for iid in self.iids_G_common_not_rated]
                predictions_u = self._rs.test(testset)
                predictions_i = self._rs2.test(testset)
                #print(len(predictions_u))
                #print(predictions_u)
                #print(len(predictions_i))
                #print(predictions_i)

                self.G_member_predictions[member] = (predictions_u, predictions_i)

            # Create a DF (uid x iid) having rij`
            # self.G_predicted_df = pd.DataFrame(columns=list(self.iids_G_common_not_rated))
            self.G_predicted_df = pd.DataFrame(columns=list(self._group_unrated_items[i]))
            self.G_predicted_df['members'] = sorted(list(G))
            self.G_predicted_df.set_index('members', inplace=True)
            # print(G_predicted_df)

            for member in G:
                predictions_u = self.G_member_predictions[member][0]
                predictions_i = self.G_member_predictions[member][1]
                for pred_u,pred_i in zip(predictions_u, predictions_i):
                    iid = pred_u.iid
                    pred_u = pred_u.est
                    pred_i = pred_i.est
                    mean_pred = (p.alpha * pred_u) + ((1-p.alpha) * pred_i)
                    #print("member:{}, iid:{}, pred_u:{}, pred_i:{}, mean_pred:{}".format(member, iid, pred_u, pred_i, mean_pred))
                    self.G_predicted_df.at[member, iid] = mean_pred

            #print(self.G_predicted_df)
            self._G_member_predictions_dfs.append(self.G_predicted_df)


    def get_cb_members_predictions_df(self):
        # For each group member predict the rating for each movie that the Group hasn't rated
        ###self.G_member_predictions_list = []
        for i, G in enumerate(self._groups):
            self.G_member_predictions = {}
            print("=== Predictions for group {} ===".format(i + 1))
            print("To predict:", len(self._group_unrated_items[i]))
            for member in G:
                #print(member)
                testset = self._group_unrated_items[i]
                ##print(testset)
                # predictions is a Pandas series
                predictions = self._rs.test(member, testset)
                self.G_member_predictions[member] = predictions

            # Create a DF (uid x iid) having rij`
            # self.G_predicted_df = pd.DataFrame(columns=list(self.iids_G_common_not_rated))
            self.G_predicted_df = pd.DataFrame(columns=list(self._group_unrated_items[i]))
            self.G_predicted_df['members'] = sorted(list(G))
            self.G_predicted_df.set_index('members', inplace=True)
            #print(self.G_predicted_df)

            for member in G:
                predictions = self.G_member_predictions[member] # This is a pandas series
                iids = list(predictions.index)
                print("iids order")
                print(iids)
                ratings = list(predictions.values)
                for iid,rating in zip(iids, ratings):
                    #print("member:{}, iid:{}, pred:{}".format(member, iid, rating))
                    self.G_predicted_df.at[member, iid] = rating

            #print(self.G_predicted_df)
            self._G_member_predictions_dfs.append(self.G_predicted_df)
            ###self.G_member_predictions_list.append(self.G_member_predictions)


    def get_hybrid_members_predictions_df(self):
        print("=== Hybrid predictions ===")
        # For each group member predict the rating for each movie that the Group hasn't rated
        self.G_member_predictions = {}
        for i, G in enumerate(self._groups):
            print("=== Predictions for group {} ===".format(i + 1))
            print("To predict:", len(self._group_unrated_items[i]))
            for member in G:
                testset = [[member, iid, 3.0] for iid in self._group_unrated_items[i]]
                testset_cb = self._group_unrated_items[i]
                predictions_svd = self._rs.test(testset)            # Surprise object
                predictions_cb = self._rs2.test(member, testset_cb) # Series
                #print(len(predictions_svd))
                #print(predictions_svd)
                #print(len(predictions_cb))
                #print(predictions_cb)

                self.G_member_predictions[member] = (predictions_svd, predictions_cb)

            # Create a DF (uid x iid) having rij`
            # self.G_predicted_df = pd.DataFrame(columns=list(self.iids_G_common_not_rated))
            self.G_predicted_df = pd.DataFrame(columns=list(self._group_unrated_items[i]))
            self.G_predicted_df['members'] = sorted(list(G))
            self.G_predicted_df.set_index('members', inplace=True)
            #print(self.G_predicted_df)

            for member in G:
                #print("~~~ Member {} ~~~".format(member))
                predictions_svd = self.G_member_predictions[member][0]
                predictions_cb = self.G_member_predictions[member][1]
                for pred_svd in (predictions_svd):
                    iid = pred_svd.iid
                    pred_svd = pred_svd.est
                    #print("pred_svd", pred_svd)
                    pred_cb = predictions_cb.loc[iid]
                    #print("pred_cb", pred_cb)
                    mean_pred = (p.alpha * pred_svd) + ((1-p.alpha) * pred_cb)
                    #print("member:{}, iid:{}, pred_svd:{}, pred_cb:{}, mean_pred:{}".format(member, iid, pred_svd,
                    #                                                                        pred_cb, mean_pred))
                    self.G_predicted_df.at[member, iid] = mean_pred

            #print(self.G_predicted_df)
            self._G_member_predictions_dfs.append(self.G_predicted_df)


    def get_ncf_members_predictions_df(self):
        print("=== NCF predictions ===")
        # For each group member predict the rating for each movie that the Group hasn't rated
        ###self.G_member_predictions_list = []
        for i, G in enumerate(self._groups):
            self.G_member_predictions = {}
            print("=== Predictions for group {} ===".format(i + 1))
            print("To predict:", len(self._group_unrated_items[i]))
            for member in G:
                #print("Member:",member)
                testset_list = list(self._group_unrated_items[i])
                print(testset_list)
                ##print("type of testset:", type(testset))
                testset = [x for x in testset_list if x < 7100]
                print(testset)
                # Create tensors
                t_member = torch.LongTensor([member])
                t_testset = torch.LongTensor(testset)

                predictions = self._rs.model(t_member, t_testset).tolist()
                predictions = zip(testset, predictions)
                predictions = [prediction for prediction in predictions]
                #print(predictions)
                self.G_member_predictions[member] = predictions

            # Create a DF (uid x iid) having rij`
            # self.G_predicted_df = pd.DataFrame(columns=list(self.iids_G_common_not_rated))
            self.G_predicted_df = pd.DataFrame(columns=list(self._group_unrated_items[i]))
            self.G_predicted_df['members'] = sorted(list(G))
            self.G_predicted_df.set_index('members', inplace=True)
            # print(self.G_predicted_df)

            for member in G:
                predictions = self.G_member_predictions[member]  # This is a pandas series
                #iids = list(predictions.index)
                #print("iids order")
                #print(iids)
                #ratings = list(predictions.values)
                for prediction in predictions:
                    iid = prediction[0]
                    rating = prediction[1]
                    #print("member:{}, iid:{}, pred:{}".format(member, iid, rating))
                    self.G_predicted_df.at[member, iid] = rating

            #print(self.G_predicted_df)
            self._G_member_predictions_dfs.append(self.G_predicted_df)
            ###self.G_member_predictions_list.append(self.G_member_predictions)



    def get_members_predictions_df(self):

        print("Get member predictions for model {}".format(self._name))
        if self._name == 'iucf':
            self.get_iucf_members_predictions_df()

        elif self._name == 'cb':
            self.get_cb_members_predictions_df()

        elif self._name == "hybrid":
            self.get_hybrid_members_predictions_df()

        elif self._name == "ncf":
            self.get_ncf_members_predictions_df()
            #print("exiting at GRS_ratings_agg.get_members_predictions_df")
            #sys.exit(4)


        else:
            # SVD, UBCF, IBCF
            # For each group member predict the rating for each movie that the Group hasn't rated
            self.G_member_predictions = {}
            for i, G in enumerate(self._groups):
                #print("=== Predictions for group {} ===".format(i+1))
                #print("To predict:",len(self._group_unrated_items[i]))
                for member in G:
                    testset = [[member, iid, 3.0] for iid in self._group_unrated_items[i]]
                    #testset = [[member, iid, 3.0] for iid in self.iids_G_common_not_rated]
                    predictions = self._rs.test(testset)
                    # print(len(predictions))

                    self.G_member_predictions[member] = predictions

                # Create a DF (uid x iid) having rij`
                #self.G_predicted_df = pd.DataFrame(columns=list(self.iids_G_common_not_rated))
                self.G_predicted_df = pd.DataFrame(columns=list(self._group_unrated_items[i]))
                self.G_predicted_df['members'] = sorted(list(G))
                self.G_predicted_df.set_index('members', inplace=True)
                # print(G_predicted_df)

                for member in G:
                    predictions = self.G_member_predictions[member]
                    for pred in predictions:
                        self.G_predicted_df.at[member, pred.iid] = pred.est

                #print(self.G_predicted_df)
                self._G_member_predictions_dfs.append(self.G_predicted_df)

        #print(self._G_member_predictions_dfs)


# --------------------------------------------
    def group_recommendations(self, function):
        self._G_recommendations_list = []
        print("~~~ Function {}~~~".format(function))
        for i, G in enumerate(self._groups):
            print("Group:",i, "members:",G)

            if function.upper() == "POP":
                print("G recommendation using:",function)
                G_recommendations = self.agg.agg_G_Popular(p.NUM_ITEMS_TO_RECOMMEND, self._dataset,
                                                           self._group_unrated_items[i])
                print("Recommendations for G using Popularity agg function:", G_recommendations)
                self._func_recommendations['POP'].append(G_recommendations)


            elif function.upper() == "AVG":
                print("G recommendation using:", function)

                G_recommendations = self.agg.agg_G_avg(p.NUM_ITEMS_TO_RECOMMEND,
                                                       self._G_member_predictions_dfs[i])
                print("Recommendations for G using Average agg function:", G_recommendations)
                self._func_recommendations['AVG'].append(G_recommendations)

            elif function.upper() == "LM":
                print("G recommendation using:", function)
                G_recommendations = self.agg.agg_G_LM(p.NUM_ITEMS_TO_RECOMMEND,
                                                      self._G_member_predictions_dfs[i])
                print("Recommendations for G using LM agg function:", G_recommendations)
                self._func_recommendations['LM'].append(G_recommendations)


            elif function.upper() == "MP":
                print("G recommendation using:", function)
                G_recommendations = self.agg.agg_G_MP(p.NUM_ITEMS_TO_RECOMMEND,
                                                      self._G_member_predictions_dfs[i])
                print("Recommendations for G using MP agg function:", G_recommendations)
                self._func_recommendations['MP'].append(G_recommendations)

            elif function.upper() == "MUL":
                print("G recommendation using:", function)
                G_recommendations = self.agg.agg_G_Mult(p.NUM_ITEMS_TO_RECOMMEND,
                                                        self._G_member_predictions_dfs[i])
                print("Recommendations for G using Multiplicative agg function:", G_recommendations)
                self._func_recommendations['MUL'].append(G_recommendations)

            elif function.upper() == "BC":
                print("G recommendation using:", function)
                G_recommendations = self.agg.agg_G_Borda(p.NUM_ITEMS_TO_RECOMMEND,
                                                         self._G_member_predictions_dfs[i])
                print("Recommendations for G using Borda Count agg function:", G_recommendations)
                self._func_recommendations['BC'].append(G_recommendations)

            elif function.upper() == "APP":
                print("G recommendation using:", function)
                G_recommendations = self.agg.agg_G_Approval(p.NUM_ITEMS_TO_RECOMMEND,
                                                            self._G_member_predictions_dfs[i])
                print("Recommendations for G using Approval agg function:", G_recommendations)
                self._func_recommendations['APP'].append(G_recommendations)

            elif function.upper() == "MRP":
                print("G recommendation using:", function)
                G_recommendations = self.agg.agg_G_MRP(p.NUM_ITEMS_TO_RECOMMEND,
                                                       self._G_members_has_rated_number_per_group[i],
                                                       self._G_member_predictions_dfs[i], G)
                print("Recommendations for G using MRP agg function:", G_recommendations)
                self._func_recommendations['MRP'].append(G_recommendations)

            elif function.upper() == "AWM":
                print("G recommendation using:", function)
                G_recommendations = self.agg.agg_G_AWM(p.NUM_ITEMS_TO_RECOMMEND, self._G_member_predictions_dfs[i])
                print("Recommendations for G using AWM agg function:", G_recommendations)
                self._func_recommendations['AWM'].append(G_recommendations)

            elif function.upper() == "ADD":
                print("G recommendation using:", function)
                G_recommendations = self.agg.agg_G_Add(p.NUM_ITEMS_TO_RECOMMEND, self._G_member_predictions_dfs[i])
                print("Recommendations for G using ADD agg function:", G_recommendations)
                self._func_recommendations['ADD'].append(G_recommendations)

            else:
                print("[ERROR] Not a valid option...")
                sys.exit(1)

            self._G_recommendations_list.append(G_recommendations)
        print(self._G_recommendations_list)
        self._G_recommendations_by_func[function].append(self._G_recommendations_list)
        print(self._G_recommendations_by_func)

        #sys.exit(8)

        #for func, G_rec in self._func_recommendations.items():
        #    self.eval.get_evaluations(func, G_rec[0], self.G_predicted_df, G)

        #self.eval.view_results()
        #self.eval.append_group_results()


        ##self.eval.get_avg_group_results()
        ##self.eval.view_group_results()
        #self.eval.save_group_results_csv()

    def get_G_recommendations_list(self):
        """

        """
        return self._G_recommendations_list

    def view_func_recommendations_for_G(self, function):
        """

        """
        recommendations_lists = self._G_recommendations_by_func[function]
        for i, G in enumerate(self._groups):
            print("Recommendations for group {} using function {}".format(i+1, function))
            print(recommendations_lists[0][i])



    def save_GRS_results(self):
        '''

        'group_size'_'group_type'_'agg_strategy'_'agg_function'_'dataset'_'metric'.csv
        '''
        self.eval.save_group_results_csv(self.group_size, self.group_type, self._agg_strategy,
                                         self._functions, self._dataset_name, self._metric)


def get_data_clusters(dataset):
    data = sd.get_pivot_table(dataset)
    cluster_labels = sd.form_clusters(data)
    print('Total labels: {}'.format(len(cluster_labels)))
    data['cluster'] = cluster_labels

    return data

if __name__ == "__main__":
    item_category = 'auto'
    ratings_filename = "../../data/" + item_category.upper() + "/ratings.csv"
    results_filename = "../../results/" + item_category.upper() + "/ratings.csv"
    metric = "hr"

    grs = GRS('svd', 'all', item_category)
    grs.load_data_and_model(ratings_filename)
    grs.form_groups(p.NUM_GROUPS, p.NUM_MEMBERS, size='s')
    grs.view_groups()
    grs.recommend_for_group_members()
    grs.set_metric(metric)
    grs.save_GRS_results()