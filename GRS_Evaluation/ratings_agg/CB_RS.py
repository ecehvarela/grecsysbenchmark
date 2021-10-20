import pandas as pd
#from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import sys
import surprise
import argparse
import numpy as np
import pickle
import GRS_Evaluation.ratings_agg.Parameters as p

class CB(object):

    def __init__(self, name):
        self._name = name
        self._data = None

    def get_name(self):
        '''
            Return the model name
        '''
        return self._name

    def get_data(self):
        '''

        '''
        return self._data

    def fit(self, _data, tfid_matrix_df, cosine_sim_matrix_df):
        self._data = _data
        self.tfid_matrix_df = tfid_matrix_df
        self.cosine_sim_matrix_df = cosine_sim_matrix_df
        self.generate_user_pref_matrix()
        self.generate_user_profile_matrix()
        self.generate_active_user_pref_matrix()
        self.get_item_avg_ratings()
        self.get_user_avg_ratings()
        self.get_pref_ratings()

        ##self._algo = [self._active_user_pref_matrix, self._item_avg_ratings]

        ##print("user pref",self._data_user_pref_bin_matrix.shape)
        ##print("user prof",self._data_user_prof_matrix.shape)
        ##print("item prof",self.tfid_matrix_df.shape)
        ##print("active user pref", self._active_user_pref_matrix.shape)

        #recommendations  = self.predict(3, 1784)
        #print(recommendations)

    def get_pref_ratings(self):
        print("Get Pref Ratings")
        #1. mult pref X item_avg_ratings
        self._active_user_ratings_df = self._active_user_pref_matrix.mul(self._item_avg_ratings,
                                                                         axis='columns')
        #2. add ratings + user_avg_ratings
        self._active_user_ratings_df = self._active_user_ratings_df.add(self._user_avg_ratings,
                                                                        axis="index")

        #print(self._active_user_ratings_df.head())
        #print(self._active_user_ratings_df.tail())
        print(self._active_user_ratings_df.shape)
        #print(self._active_user_ratings_df.loc[1].sort_values(ascending=False))


    def predict(self, uid, iid, r=3):
        # get the top-k items
        user_rec = self._active_user_ratings_df.loc[uid].sort_values(ascending=False)#[:p.NUM_ITEMS_TO_RECOMMEND]
        #print(user_rec)
        #print(user_rec.shape)
        #user_rec_indices = list(user_rec.index)
        #print(user_rec_indices)
        # get the avg. ratings for the top-k items
        #user_rec_avg_rating = self._item_avg_ratings[user_rec_indices]
        #print(user_rec_avg_rating)
        return user_rec.loc[iid]

    def test(self, uid, to_predict):
        #print("CB test")
        #print("member: {}".format(uid))
        ##print(to_predict)
        ##print(len(to_predict))

        user_rec = self._active_user_ratings_df.loc[uid]
        user_rec = user_rec.loc[to_predict]
        ###Revisar####user_rec = user_rec.sort_values(ascending=False)
        #print(user_rec.shape)
        #user_rec_indices = list(user_rec.index)
        #user_rec_avg_rating = self._item_avg_ratings[user_rec_indices]
        #print(user_rec_avg_rating)

        #return user_rec_avg_rating
        #print(user_rec)
        #print(user_rec.shape)

        return user_rec


    def get_item_avg_ratings(self):
        print("Get Item Avg Ratings")
        #print(self._data_user_pref_matrix.head())
        self._item_avg_ratings = self._data_user_pref_matrix.mean(axis=0, skipna=True)
        print(self._item_avg_ratings.shape)
        #print(self._item_avg_ratings)
        #print(type(self._item_avg_ratings))

    def get_user_avg_ratings(self):
        print("Get User Avg Ratings")
        self._user_avg_ratings = self._data_user_pref_matrix.mean(axis=1, skipna=True)
        print(self._user_avg_ratings.shape)
        #print(self._user_avg_ratings.head())


    def generate_user_pref_matrix(self):
        print("Generate User Pref Matrix")
        # pivot table
        #print(self._data)

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
        print(self._data_user_prof_matrix.shape)


    def generate_active_user_pref_matrix(self):
        print("Active User Pref Matrix")
        self._active_user_pref_matrix = cosine_similarity(self._data_user_prof_matrix, self.tfid_matrix_df,
                                                          dense_output=True)

        iid_indices = list(self.tfid_matrix_df.index)
        uid_indices = list(self._data_user_pref_bin_matrix.index)
        self._active_user_pref_matrix = pd.DataFrame(self._active_user_pref_matrix, index=uid_indices, columns=iid_indices)

        #print(self._active_user_pref_matrix)
        print(self._active_user_pref_matrix.shape)
