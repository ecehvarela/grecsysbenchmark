'''
Class for the individual Recommender Systems Models

# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

'''

import pandas as pd
#from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import sys
import surprise
import argparse
import numpy as np
import pickle
import torch
import GRS_Evaluation.ratings_agg.Parameters as p
import GRS_Evaluation.ratings_agg.CB_RS as CB
from fastai.collab import CollabDataBunch, collab_learner
from fastai.basic_train import load_learner

class Model(object):

    def __init__(self, name):
        self._name = name
        self._data = None
        self._algo = None

    def get_name(self):
        '''
            Return the model name
        '''
        return self._name


    def load_matrices_data(self, matrices_filename):

        object_filename = matrices_filename

        file = open(object_filename, 'rb')
        data = pickle.load(file)
        file.close()

        return data

    def get_matrices_data(self, matrices_filename):
        data = self.load_matrices_data(matrices_filename)
        print("Loading matrices in: {}".format(matrices_filename))
        self.tfid_matrix_df = data[0]
        self.cosine_sim_matrix_df = data[1]


    def load_data(self, ratings_filename):
        '''
            Read the data filename
            Return the model dataset
        '''
        print("Loading ratings from :'{}'".format(ratings_filename))
        self._data = pd.read_csv(ratings_filename)


    def get_data(self):
        '''

        '''
        return self._data

    def get_model(self):
        '''for group_type in p.group_type:
            print("\t Using Group type: {}".format(group_type))
            for group_size in p.group_size:
                print("\t\t Using Group size: {}".format(group_size))
                for function in p.agg_functions:
                    print("\t\t\t Using function: {}".format(function))
                    print("\t\t\t For each of the {} groups".format(ng))
                    for metric in p.metrics:
                        print("\t\t\t\t\t Calculate {}".format(metric))
                        print("\t\t\t\t\t Average {} results".format(metric))
                        print("\t\t\t\t\t Save results as {}".format(data+"_"+group_type+"_"+group_size+
                                                                     "_"+function+"_"+metric+".csv"))
                        print("\t\t\t\t\t -------------------")


            Return the trained model
        '''
        return self._algo


    def save_model(self, model_file):
        '''
            Save the trained model
        '''

        if self._name == "ncf":
            self._algo.export(model_file)

        else:
            joblib.dump(self._algo, model_file)

        print("Model saved as {}".format(model_file))


    def load_model(self, model_file):
        '''
            Load the saved trained model
        '''

        if self._name == "ncf":
            self._algo = load_learner(path='.', file=model_file)
        else:
            self._algo = joblib.load(model_file)

        print("Model in {} loaded".format(model_file))
        return self._algo

    def train_model(self):

        m_name = self.get_name()
        print("The model name is {}".format(m_name))
        lower_rating, upper_rating = get_ratings_bounds(self._data)
        print("Review range: {} to {}".format(lower_rating, upper_rating))
        data = get_data_for_surprise(self._data, lower_rating, upper_rating)

        if m_name == "svd":
            # defining the model
            self._algo = surprise.SVDpp(n_factors=p.svd_factors)
            # training the model
            print("Training model {}...".format(m_name.upper()))
            self._algo.fit(data.build_full_trainset())


        elif m_name == "ubcf":
            #print("The model name is {}".format(m_name))
            #lower_rating, upper_rating = get_ratings_bounds(self._data)
            #print("Review range: {} to {}".format(lower_rating, upper_rating))
            #data = get_data_for_surprise(self._data, lower_rating, upper_rating)
            # defining the model
            sim_options = {'name': 'cosine'}
            self._algo = surprise.KNNWithMeans(k=50, sim_options=sim_options, user_based=True)
            print("Training model {}...".format(m_name.upper()))
            self._algo.fit(data.build_full_trainset())


        elif m_name == 'ibcf':
            #print("The model name is {}".format(m_name))
            #lower_rating, upper_rating = get_ratings_bounds(self._data)
            #print("Review range: {} to {}".format(lower_rating, upper_rating))
            #data = get_data_for_surprise(self._data, lower_rating, upper_rating)
            # defining the model
            sim_options = {'name': 'cosine'}
            self._algo = surprise.KNNWithMeans(k=50, sim_options=sim_options, user_based=False)
            print("Training model {}...".format(m_name.upper()))
            self._algo.fit(data.build_full_trainset())

        elif m_name == 'cb':
            # generate user profile
            # get user preference matrix
            print("This is model {}".format(m_name.upper()))
            ##self.train_cb_algo()
            self._algo = CB.CB(m_name)
            self._algo.fit(self._data, self.tfid_matrix_df, self.cosine_sim_matrix_df)

        elif m_name == 'ncf':
            print("This is model {}".format(m_name.upper()))
            data_ncf = CollabDataBunch.from_df(self._data, valid_pct=0)
            self._algo = collab_learner(data_ncf, n_factors=p.ncf_factors, y_range=(0, upper_rating+1), wd=.1)
            print("Training model {}...".format(m_name.upper()))
            self._algo.fit_one_cycle(p.ncf_epochs, p.ncf_lr)

        else:
            print("This model is unknown!")
            sys.exit(8)


    def dump(self, filename):
        surprise.dump.dump(filename, algo=self._algo)


    def train_cb_algo(self):
        self.generate_user_pref_matrix()
        self.generate_user_profile_matrix()
        self.generate_active_user_pref_matrix()
        self.get_item_avg_ratings()

        self._algo = [self._active_user_pref_matrix, self._item_avg_ratings]

        ##print("user pref",self._data_user_pref_bin_matrix.shape)
        ##print("user prof",self._data_user_prof_matrix.shape)
        ##print("item prof",self.tfid_matrix_df.shape)
        ##print("active user pref", self._active_user_pref_matrix.shape)

        #recommendations  = self.predict_cb_algo(3)
        #print(recommendations)

    def predict_cb_algo(self, uid):
        # get the top-k items
        user_rec = self._active_user_pref_matrix.loc[uid].sort_values(ascending=False)[:p.NUM_ITEMS_TO_RECOMMEND]
        ##print(user_rec)
        #print(user_rec.shape)
        user_rec_indices = list(user_rec.index)
        #print(user_rec_indices)
        # get the avg. ratings for the top-k items
        user_rec_avg_rating = self._item_avg_ratings[user_rec_indices]
        #print(user_rec_avg_rating)
        return user_rec_avg_rating


    def get_item_avg_ratings(self):
        print("Get Item Avg Ratings")
        #print(self._data_user_pref_matrix.head())
        self._item_avg_ratings = self._data_user_pref_matrix.mean(axis=0, skipna=True)
        #print(self._item_avg_ratings)



    def generate_user_pref_matrix(self):
        print("Generate User Pref Matrix")
        # pivot table
        self._data_user_pref_matrix = pd.pivot_table(self._data, index='uid', columns='iid', values='rating')
        print(self._data_user_pref_matrix.shape)
        # change to binary values
        #print(self._data_user_pref_matrix.head())
        self._data_user_pref_bin_matrix = (self._data_user_pref_matrix > 0) * 1

    def generate_user_profile_matrix(self):
        print("Generate User Profile Matrix")
        self._data_user_prof_matrix = np.dot(self._data_user_pref_bin_matrix, self.tfid_matrix_df) \
                                    / np.linalg.norm(self._data_user_pref_bin_matrix) \
                                    /np.linalg.norm(self.tfid_matrix_df)
        #print(type(self._data_user_prof_matrix))
        #print(self._data_user_prof_matrix.shape)
        #print(self.__data_user_prof_matrix)
        uid_indices = list(self._data_user_pref_bin_matrix.index)
        self._data_user_prof_matrix = pd.DataFrame(self._data_user_prof_matrix, index=uid_indices)
        #print(self._data_user_prof_matrix.head())


    def generate_active_user_pref_matrix(self):
        print("Active User Pref Matrix")
        self._active_user_pref_matrix = cosine_similarity(self._data_user_prof_matrix, self.tfid_matrix_df,
                                                          dense_output=True)

        iid_indices = list(self.tfid_matrix_df.index)
        uid_indices = list(self._data_user_pref_bin_matrix.index)
        self._active_user_pref_matrix = pd.DataFrame(self._active_user_pref_matrix, index=uid_indices, columns=iid_indices)
        #print(self._active_user_pref_matrix)


def get_ratings_bounds(data):
    lower_rating = data['rating'].min()
    upper_rating = data['rating'].max()

    return lower_rating, upper_rating

def get_data_for_surprise(data, lower_rating, upper_rating):
    reader = surprise.Reader(rating_scale=(lower_rating, upper_rating))
    data = surprise.Dataset.load_from_df(data, reader)

    return data





#####################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Main file creating models")
    # dataset
    parser.add_argument("-dataset", required=True, type=str,
                        help="Name of the dataset, e.g. op, bea, dm")
    # data folder
    parser.add_argument("-path", required=True, type=str,
                        help="Path for the data folder")
    # RS model
    parser.add_argument("-rs", required=True, type=str,
                        choices=["ibcf", "ubcf", "iucf", "cb", "hybrid", "svd", "ncf", "camra2011"],
                        help="Recommender system model "
                             "[IBCF]: Item-based CF, "
                             "[UBCF]: User-based CF, "
                             "[IUCF]: IICF+UUCF, "
                             "[CB]: Conent Based, "
                             "[Hybrid]: hybrid,"
                             "[SVD]: SVD++,"
                             "[NCF]: Neural CF")

    args = parser.parse_args()

    dataset = args.dataset
    data_path = args.path
    model_list = [args.rs]
    #item_category = 'baby'
    #"../../data
    ratings_filename = data_path.lower() + "/" + dataset.upper() + "/ratings.csv"

    matrices_filename = data_path.lower() + "/" + dataset.upper() + "/matrices.obj"

    ###
    # Create the model
    #model_list = ['svd', 'uucf', 'iicf']
    #model_list = ['svd']
    for model in model_list:
        rs = Model(model)
        print(rs.get_name())
        print("Factors", p.ncf_factors)
        rs.load_data(ratings_filename)
        print(rs.get_data().head())
        # rs.get_matrices_data(matrices_filename) #No categories/matrices for CAMRa2011

        rs.train_model()
        rs.save_model("models/"+dataset+"_"+model+".model")

        print()
        uid = 1
        iid = 1
        r = 5.0

        if (rs.get_name() == 'ncf'):
            uid = [uid]
            iid = [iid]
            tensor_users = torch.LongTensor(uid)
            tensor_items = torch.LongTensor(iid)
            print(rs._algo.model(tensor_users, tensor_items).tolist())

        else:
            print(rs._algo.predict(uid, iid, r))

        #rs.dump("models/"+dataset+"_"+model+".model2")
