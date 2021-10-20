import GRS_Evaluation.profiles_agg.Model as m
import GRS_Evaluation.profiles_agg.Parameters as p
import GRS_Evaluation.profiles_agg.AggFunctions as af
import GRS_Evaluation.Evaluator_profiles as eval
import GRS_Evaluation.sim_disim_groups as sd
import GRS_Evaluation.profiles_agg.item_categories_extraction_profiles as mat

import gc
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

        self.cosine_sim_matrix_df = None
        self.tfid_matrix_df = None

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

    def get_matrices(self, filename, out_filename, save):
        '''

        '''
        print(self._dataset_name)
        if self._dataset_name == "ml":
            print("Loading matrices for MovieLens")
            data = mat.load_object_ML(out_filename)
        else:
            data = mat.get_matrices(filename, out_filename, save)
        self.tfid_matrix_df = data[0]
        self.cosine_sim_matrix_df = data[1]

        ##print("self.tfidf")
        ##print(self.tfid_matrix_df)
        ##print("self.cosine")
        ##print(self.cosine_sim_matrix_df)


    def load_matrices(self, matrices_filename):
        '''

        '''

        print("Loading matrices in: {}".format(matrices_filename))
        data = m.Model(self._name).load_matrices_data(matrices_filename)
        self.tfid_matrix_df = data[0]
        self.cosine_sim_matrix_df = data[1]


    def add_virtual_user_to_ratings(self, vu_id, vu_ratings):
        vu_ratings = vu_ratings.reset_index(drop=False)
        if vu_ratings.shape[0] < 1:
            print("No ratings for this user")
            sys.exit(4)

        vu_ratings['uid'] = vu_id
        vu_ratings.columns = ['iid', 'rating', 'uid']
        vu_ratings = vu_ratings[['uid', 'iid', 'rating']]
        #print(vu_ratings.head())
        #print(vu_ratings.shape)
        self._dataset_vu = pd.concat([self._dataset_vu, vu_ratings], ignore_index=True)
        #print(self._dataset)
        print("just added user: {}".format(vu_id))
        print("max user id in dataset:",self._dataset_vu['uid'].max())
        ##print(self._dataset_vu.tail(15))


    def get_next_virtual_user_id(self, group_i):
        #print("The highest user id is:", max(self._uids))
        next_virtual_user_id = max(self._uids) + 1 + group_i
        print("This group's virtual user id", next_virtual_user_id)
        return  next_virtual_user_id

    def generate_virtual_user_pivot_dataset(self):
        '''

        '''
        ##print("DATASET~")
        ##print(self._dataset)
        self._dataset = self._dataset.reset_index(drop=True)

        self._pivot_df = pd.pivot(self._dataset, index='uid', columns='iid', values='rating')
        ##print("all pivot")
        ##print(self._pivot_df)

        #self._v_users_pivot_df = self._pivot_df.loc[self._v_users]
        #self._v_users_pivot_df = self._v_users_pivot_df.dropna(how='all', axis=1)

        ##print("exiting in GRS_ratings_agg.generate_virtual_user_pivot_dataset()")
        ##sys.exit(3)

    def generate_virtual_user_profile(self, function):
        # Make a copy of the initial _dataset
        self._dataset = self._dataset_original.copy(deep=True)

        print("_dataset entering here:",self._dataset.shape)
        self._dataset_vu = self._dataset.copy(deep=True)
        print(type(self._dataset_vu))
        print(list(self._dataset_vu.columns))
        print("unique users: {}".format(self._dataset_vu['uid'].nunique()))
        print("maximum user id: {}".format(self._dataset_vu['uid'].max()))

        self._v_users = []
        self._v_user_rated_items = {}

        print("For all the groups, get the VIRTUAL USER profile")
        #print(self._dataset)

        # Get the pivot dataset
        user_items_pivot_df = pd.pivot_table(self._dataset, index='uid', columns='iid',
                                            values='rating')
        #user_items_pivot_df.fillna(0, inplace=True)

        for i, G in enumerate(self._groups):
            # make a copy of dataset
            #print(self._dataset.shape)
            print("Group {}: {}".format(i+1, G))
            virtual_user_id = self.get_next_virtual_user_id(i)
            G_ratings = user_items_pivot_df.loc[G]
            G_ratings.dropna(axis='columns', how='all', inplace=True)
            G_ratings.fillna(0, inplace=True)
            #print(G_ratings)

            # Get the agg profile using the function
            vu_rating_profile = self.get_virtual_user_profile(virtual_user_id, function, G_ratings)
            #print(vu_rating_profile)

            # add the profile to the new ratings dataset
            print("Adding user {}'s ratings to dataset".format(virtual_user_id))
            self.add_virtual_user_to_ratings(virtual_user_id, vu_rating_profile)
            print()

            self._v_user_rated_items[virtual_user_id] = list(set(vu_rating_profile.index))
            self._v_users.append(virtual_user_id)

            print()

        # Update _dataset with _dataset_vu
        print("exiting, unique users: {}".format(self._dataset_vu['uid'].nunique()))
        print("exiting,maximum user id: {}".format(self._dataset_vu['uid'].max()))
        self._dataset = self._dataset_vu
        self._dataset = self._dataset.sort_values(by='iid')
        print("_dataset exiting here:",self._dataset.shape)
        print("unique users: {}".format(self._dataset['uid'].nunique()))
        print("maximum user id {}:".format(self._dataset['uid'].max()))
        print(self._v_user_rated_items)
        #self._dataset.to_csv("TESTS_DATASET.csv", index=False)


    def create_model(self):
        print("This will train a {} model".format(self._name))

        if "IUCF" == self._name.upper():
            print("Create a model for 'ubcf'")
            self._name = 'ubcf'
            #print(self._name)
            rs = m.Model(self._name)
            self._rs = rs.train_model(self._dataset, self.tfid_matrix_df, self.cosine_sim_matrix_df)

            print("Create a model for 'ibcf'")
            self._name = 'ibcf'
            #print(self._name)
            rs2 = m.Model(self._name)
            self._rs2 = rs2.train_model(self._dataset, self.tfid_matrix_df, self.cosine_sim_matrix_df)

            # return the name to the original
            self._name = 'iucf'
            ##print(self._name)
            del rs
            del rs2


        elif "HYBRID" == self._name.upper():
            print("Create a model for 'svd'")
            self._name = 'svd'
            # print(self._name)
            rs = m.Model(self._name)
            self._rs = rs.train_model(self._dataset, self.tfid_matrix_df, self.cosine_sim_matrix_df)

            print("Create a model for 'cb'")
            self._name = 'cb'
            # print(self._name)
            rs2 = m.Model(self._name)
            self._rs2 = rs2.train_model(self._dataset, self.tfid_matrix_df, self.cosine_sim_matrix_df)

            # return the name to the original
            self._name = 'hybrid'
            ##print(self._name)

        elif "NCF" == self._name.upper():
            print("Create a model for ncf")
            self._name = 'ncf'

            rs = m.Model(self._name)
            self._rs = rs.train_model(self._dataset, self.tfid_matrix_df, self.cosine_sim_matrix_df)

        else:
            print("Create a model for {ubcf | ibcf | svd | cb}")
            rs = m.Model(self._name)
            self._rs = rs.train_model(self._dataset, self.tfid_matrix_df, self.cosine_sim_matrix_df)



    def delete_model(self):
        print("Deleting previous models, if they exist ...")
        print(gc.isenabled())
        print(gc.collect())
        for item in gc.garbage:
            print("\t",item)

        try:
            self._rs = None
            del self._rs

            self._rs2 = None
            del self._rs2
        except:
            print("Not found...")


    def delete_dataset(self):
        self._dataset = None

    def load_data_and_model(self, ratings_filename, model_filename):
        '''

        '''
        # load the dataset
        rs = m.Model(self._name)
        rs.load_data(ratings_filename)
        self._dataset = rs._data
        self._dataset_original = self._dataset.copy(deep=True)

        print("Dataset when loaded....")
        print(self._dataset.shape)

        # Get the list of uids
        self._uids = self._dataset['uid'].unique()
        print("There are {} users".format(len(self._uids)))
        print("The mininum users id:", self._uids.min())
        print("The maximum users id:", self._uids.max())
        self._iids = self._dataset['iid'].unique()
        print("Total items:", len(self._iids))
        print("The mininum items id:", self._iids.min())
        print("The maximum items id:", self._iids.max())

        del rs

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

        group_member_path = "../data/CAMRA2011/groupMember.txt"

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

    def get_virtual_users(self):
        '''
            Return the list of virtual users
        '''
        return self._v_users


    def view_groups(self):
        '''
            Display the groups and their members
        '''
        for i, G in enumerate(self._groups):
            # print("Group", i+1 ,":", ",".join([str(e) for e in G.tolist()]))
            print("Group", i + 1, ":", ",".join([str(e) for e in G]))

    def items_rated_for_group_member(self, member):
        print("Getting iid list for member:", member)
        iids_rated = self._dataset.loc[self._dataset['uid'] == member, 'iid']
        ##print("\t items rated by user {}: {}".format(member, len(iids_rated)))
        return iids_rated


    def get_all_user_items_not_rated(self):
        self._user_not_rated = {}
        #print(self._dataset.head())

        #print("HERE in get_all_user_items_not_rated")
        ##print(self._uids)
        ##print(len(self._uids))
        ##print(self._v_users)
        ##print(len(self._v_users))
        all_uids = list(self._uids) + list(self._v_users)
        ##print(all_uids)
        ##print(len(all_uids))

        #for user in list(self._pivot_df.index):
        for user in list(all_uids):
            rated_iids = self._dataset[self._dataset['uid'] == user]['iid']
            rated_iids = set(rated_iids.unique())
            #print(rated_iids)
            #print(len(rated_iids))

            unrated_items = list( set(self._iids) - rated_iids )
            #print(unrated_items)
            #print(len(unrated_items))
            self._user_not_rated[user] = unrated_items

        ##print(self._user_not_rated[1])


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
        self._all_predictions = {}

        group_members = set()
        for G in self._groups:
            g_members = set([member for member in G])
            group_members = group_members.union(g_members)

        for v_user in self._v_users:
            group_members.add(v_user)

        print("All users:", group_members)

        for user in group_members:
            items_not_rated = self._user_not_rated[user]
            testset = [[user, iid, 3.0] for iid in items_not_rated]
            predictions_u = self._rs.test(testset)
            predictions_i = self._rs2.test(testset)
            self._all_predictions[user] = (predictions_u, predictions_i)

        self._user_predicted_df = pd.DataFrame(columns=list(self._iids))
        self._user_predicted_df['members'] = sorted(group_members)
        self._user_predicted_df.set_index('members', inplace=True)
        # print(G_predicted_df)

        for user in group_members:
            predictions_u = self._all_predictions[user][0]
            predictions_i = self._all_predictions[user][1]
            for pred_u, pred_i in zip(predictions_u, predictions_i):
                iid = pred_u.iid
                pred_u = pred_u.est
                pred_i = pred_i.est
                mean_pred = (p.alpha * pred_u) + ((1 - p.alpha) * pred_i)
                # print("member:{}, iid:{}, pred_u:{}, pred_i:{}, mean_pred:{}".format(member, iid, pred_u, pred_i, mean_pred))
                self._user_predicted_df.at[user, iid] = mean_pred

        # print(self._user_predicted_df)



    def get_cb_members_predictions_df(self):

        self._all_predictions = {}

        # Get all users within groups
        group_members = set()
        for G in self._groups:
            g_members = set([member for member in G])
            group_members = group_members.union(g_members)

        # Add the virtual users
        for v_user in self._v_users:
            group_members.add(v_user)

        print("All user:", group_members)
        for user in group_members:
            #print("user:", user)
            testset= self._user_not_rated[user]
            #print(testset)
            predictions = self._rs.test(user, testset)
            # print(predictions)
            # print(len(predictions))
            self._all_predictions[user] = predictions
            # print("~~" * 50)

        # Create a DF (uid x iid) having rij`
        # self.G_predicted_df = pd.DataFrame(columns=list(self.iids_G_common_not_rated))
        self._user_predicted_df = pd.DataFrame(columns=self._iids)
        self._user_predicted_df['members'] = sorted(group_members)
        self._user_predicted_df.set_index('members', inplace=True)

        for user in group_members:
            predictions = self._all_predictions[user] # This is a pandas series
            iids = list(predictions.index)
            #print("iids order")
            #print(iids)
            ratings = list(predictions.values)
            for iid,rating in zip(iids, ratings):
                self._user_predicted_df.at[user, iid] = rating

        #print(self._user_predicted_df)
###################


#############
    def get_nfc_members_predictions_df(self):
        print("=== NFC predictions ===")
        self._all_predictions = {}

        # Get all users within groups
        group_members = set()
        for G in self._groups:
            g_members = set([member for member in G])
            group_members = group_members.union(g_members)

        # Add the virtual users
        for v_user in self._v_users:
            group_members.add(v_user)

        print("All user:", group_members)
        for user in group_members:
            # print("user", user)
            testset_list = list(self._user_not_rated[user])
            print(testset_list)
            testset = [x for x in testset_list if x < 7100]
            print(testset)
            # Create tensors
            t_member = torch.LongTensor([user])
            t_testset = torch.LongTensor(testset)

            # print("Member:",t_member)
            # for t in testset:
            #     print("Item:", t)
            #     t_testset = torch.LongTensor([t])
            #     predictions = self._rs.model(t_member, t_testset).tolist()
            #     print(predictions)
            # sys.exit("AAAQQUUUII")

            predictions = self._rs.model(t_member, t_testset).tolist()
            predictions = zip(testset, predictions)
            predictions = [prediction for prediction in predictions]
            # print(predictions)
            self._all_predictions[user] = predictions

        # Create a DF (uid x iid) having rij`
        # self.G_predicted_df = pd.DataFrame(columns=list(self.iids_G_common_not_rated))
        self._user_predicted_df = pd.DataFrame(columns=self._iids)
        self._user_predicted_df['members'] = sorted(group_members)
        self._user_predicted_df.set_index('members', inplace=True)

        for user in group_members:
            predictions = self._all_predictions[user]  # This is a pandas series
            # iids = list(predictions.index)
            # print("iids order")
            # print(iids)
            # ratings = list(predictions.values)
            for prediction in predictions:
                iid = prediction[0]
                rating = prediction[1]
                # print("member:{}, iid:{}, pred:{}".format(member, iid, rating))
                self._user_predicted_df.at[user, iid] = rating

        print(self._user_predicted_df)


    def get_hybrid_members_predictions_df(self):
        print("=== Hybrid predictions ===")
        # For each group member predict the rating for each movie that the Group hasn't rated
        self._all_predictions = {}

        group_members = set()
        for G in self._groups:
            g_members = set([member for member in G])
            group_members = group_members.union(g_members)

        for v_user in self._v_users:
            group_members.add(v_user)

        print("All users:", group_members)

        for user in group_members:
            items_not_rated = self._user_not_rated[user]
            testset_svd = [[user, iid, 3.0] for iid in items_not_rated]
            testset_cb = self._user_not_rated[user]

            predictions_svd = self._rs.test(testset_svd)
            predictions_cb = self._rs2.test(user, testset_cb)

            self._all_predictions[user] = (predictions_svd, predictions_cb)

        # Create a DF (uid x iid) having rij`
        # self.G_predicted_df = pd.DataFrame(columns=list(self.iids_G_common_not_rated))
        self._user_predicted_df = pd.DataFrame(columns=self._iids)
        self._user_predicted_df['members'] = sorted(group_members)
        self._user_predicted_df.set_index('members', inplace=True)

        for user in group_members:
            predictions_svd = self._all_predictions[user][0]
            predictions_cb = self._all_predictions[user][1]
            for pred_svd in predictions_svd:
                iid = pred_svd.iid
                pred_svd = pred_svd.est
                # print("pred_svd", pred_svd
                pred_cb = predictions_cb.loc[iid]
                # print("pred_cb", pred_cb)
                mean_pred = (p.alpha * pred_svd) + ( (1-p.alpha) * pred_cb)
                # print("member:{}, iid:{}, pred_svd:{}, pred_cb:{}, mean_pred:{}".format(member, iid, pred_svd,
                #
                self._user_predicted_df.at[user, iid] = mean_pred

        #print(self._user_predicted_df)



    def get_others_members_predictions_df(self):
        # SVD, UBCF, IBCF
        # For each group member predict the rating for each movie that the Group hasn't rated
        self._virtual_user_predictions = {}
        self._all_predictions = {}

        # Get all users within groups
        group_members = set()
        for G in self._groups:
            g_members = set([member for member in G])
            group_members = group_members.union(g_members)

        # Add the virtual users
        for v_user in self._v_users:
            group_members.add(v_user)

        print("All user:", group_members)
        for user in group_members:
            #print("user:", user)
            items_not_rated = self._user_not_rated[user]
            testset = [[user, iid, 3.0] for iid in items_not_rated]
            #print(testset)
            predictions = self._rs.test(testset)
            #print(predictions)
            #print(len(predictions))
            self._all_predictions[user] = predictions
            #print("~~" * 50)

        # Create a DF (uid x iid) having rij`
        # self.G_predicted_df = pd.DataFrame(columns=list(self.iids_G_common_not_rated))
        self._user_predicted_df = pd.DataFrame(columns=self._iids)
        self._user_predicted_df['members'] = sorted(group_members)
        self._user_predicted_df.set_index('members', inplace=True)

        for user in group_members:
            predictions = self._all_predictions[user]
            for pred in predictions:
                self._user_predicted_df.at[user, pred.iid] = pred.est

        #print(self._user_predicted_df)



    def get_all_user_recommendations(self):
        #print(self._user_predicted_df)
        users = list(self._user_predicted_df.index)
        self._user_recommendations = {}
        for user in users:
            recommendations = self._user_predicted_df.loc[user]
            recommendations = recommendations.sort_values(ascending=False)
            recommendations = recommendations[:p.NUM_ITEMS_TO_RECOMMEND]
            recommendations = list(recommendations.index)
            self._user_recommendations[user] = recommendations

    def get_test_prediction(self):
        print("TESTING!!!!!")

        print(self._rs)
        user = 1818
        testset = [1, 2]
        t_member = torch.LongTensor([user])
        t_testset = torch.LongTensor(testset)
        predictions = self._rs.model(t_member, t_testset).tolist()
        print(predictions)


    def get_all_user_predictions_df(self):
        #print("virtual users:",self._v_users)
        print("Get member predictions for model {}".format(self._name))

        if self._name == 'iucf':
            self.get_iucf_members_predictions_df()

        elif self._name == 'cb':
            self.get_cb_members_predictions_df()

        elif self._name == "hybrid":
            self.get_hybrid_members_predictions_df()

        elif self._name == "ncf":
            self.get_nfc_members_predictions_df()


        else:
            self.get_others_members_predictions_df()
            #print("exiting on GRS_ratings_agg.get_all_user_predictions_df()")
            #sys.exit(3)


    def testing(self):
        print("virtual users profile ratings")
        print(self._v_users_pivot_df)
        print("virtual users predictions")
        print(self.virtual_user_predicted_df)

        print("exiting in GRS_ratings_agg.testing")
        sys.exit(8)


# --------------------------------------------
    def get_virtual_user_profile(self, virtual_user_id, function, G_dataset):
        self._G_recommendations_list = []
        print("~~~ Function {}~~~".format(function))

        if function.upper() == "POP":
            print("Virtual User {} profile using: {}".format(virtual_user_id, function))
            v_user_profile = self.agg.agg_G_Popular_2(G_dataset, self._dataset)
            #print("Recommendations for G using Popularity agg function:", G_recommendations)
            #self._func_recommendations['POP'].append(G_recommendations)
            return v_user_profile

        elif function.upper() == "AVG":
            print("Virtual User {} profile using: {}".format(virtual_user_id, function))

            v_user_profile = self.agg.agg_G_avg(G_dataset)
            #print("Profile for virtual user {} using Average agg function:".format(virtual_user_id))
            #print(v_user_profile)
            ##self._func_recommendations['AVG'].append(v_user_profile)
            return v_user_profile


        elif function.upper() == "LM":
            print("Virtual User {} profile using: {}".format(virtual_user_id, function))

            v_user_profile = self.agg.agg_G_LM(G_dataset)

            #print("G recommendation using:", function)
            #G_recommendations = self.agg.agg_G_LM(p.NUM_ITEMS_TO_RECOMMEND,
            #                                      self._G_member_predictions_dfs[i])
            #print("Recommendations for G using LM agg function:", G_recommendations)
            #self._func_recommendations['LM'].append(G_recommendations)
            return v_user_profile


        elif function.upper() == "MP":
            print("Virtual User {} profile using: {}".format(virtual_user_id, function))

            v_user_profile = self.agg.agg_G_MP(G_dataset)

            #print("G recommendation using:", function)
            #G_recommendations = self.agg.agg_G_MP(p.NUM_ITEMS_TO_RECOMMEND,
            #                                      self._G_member_predictions_dfs[i])
            #print("Recommendations for G using MP agg function:", G_recommendations)
            #self._func_recommendations['MP'].append(G_recommendations)
            return v_user_profile

        elif function.upper() == "MUL":
            print("Virtual User {} profile using: {}".format(virtual_user_id, function))

            v_user_profile = self.agg.agg_G_Mult(G_dataset)

            #print("G recommendation using:", function)
            #G_recommendations = self.agg.agg_G_Mult(p.NUM_ITEMS_TO_RECOMMEND,
            #                                        self._G_member_predictions_dfs[i])
            #print("Recommendations for G using Multiplicative agg function:", G_recommendations)
            #self._func_recommendations['MUL'].append(G_recommendations)
            return v_user_profile

        elif function.upper() == "BC":
            print("Virtual User {} profile using: {}".format(virtual_user_id, function))

            v_user_profile = self.agg.agg_G_Borda(G_dataset, self._dataset)


            #print("G recommendation using:", function)
            #G_recommendations = self.agg.agg_G_Borda(p.NUM_ITEMS_TO_RECOMMEND,
            #                                         self._G_member_predictions_dfs[i])
            #print("Recommendations for G using Borda Count agg function:", G_recommendations)
            #self._func_recommendations['BC'].append(G_recommendations)
            return v_user_profile

        elif function.upper() == "APP":
            print("Virtual User {} profile using: {}".format(virtual_user_id, function))

            v_user_profile = self.agg.agg_G_Approval(G_dataset, self._dataset)

            #print("G recommendation using:", function)
            #G_recommendations = self.agg.agg_G_Approval(p.NUM_ITEMS_TO_RECOMMEND,
            #                                            self._G_member_predictions_dfs[i])
            #print("Recommendations for G using Approval agg function:", G_recommendations)
            #self._func_recommendations['APP'].append(G_recommendations)
            return v_user_profile

        elif function.upper() == "MRP":
            print("Virtual User {} profile using: {}".format(virtual_user_id, function))

            v_user_profile = self.agg.agg_G_MRP(G_dataset, self._dataset)

            #print("G recommendation using:", function)
            #G_recommendations = self.agg.agg_G_MRP(p.NUM_ITEMS_TO_RECOMMEND,
            #                                       self._G_members_has_rated_number_per_group[i],
            #                                       self._G_member_predictions_dfs[i], G)
            #print("Recommendations for G using MRP agg function:", G_recommendations)
            #self._func_recommendations['MRP'].append(G_recommendations)
            return v_user_profile

        elif function.upper() == "AWM":
            print("Virtual User {} profile using: {}".format(virtual_user_id, function))

            v_user_profile = self.agg.agg_G_AWM(G_dataset)

            #print("G recommendation using:", function)
            #G_recommendations = self.agg.agg_G_AWM(p.NUM_ITEMS_TO_RECOMMEND, self._G_member_predictions_dfs[i])
            #print("Recommendations for G using AWM agg function:", G_recommendations)
            #self._func_recommendations['AWM'].append(G_recommendations)
            return v_user_profile

        elif function.upper() == "ADD":
            print("Virtual User {} profile using: {}".format(virtual_user_id, function))

            v_user_profile = self.agg.agg_G_Add(G_dataset)

            #print("G recommendation using:", function)
            #G_recommendations = self.agg.agg_G_Add(p.NUM_ITEMS_TO_RECOMMEND, self._G_member_predictions_dfs[i])
            #print("Recommendations for G using ADD agg function:", G_recommendations)
            #self._func_recommendations['ADD'].append(G_recommendations)
            return v_user_profile

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
    print(" ==== HERE!!!")
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