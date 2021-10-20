'''
Class having the different rating aggregation functions

'''
import numpy as np

class AggFunctions():

    def __init__(self):
        pass

    def pick_top_k_recommendations(self, iids, top_k):
        '''
            Given a list of items, with their ratings (or scores)
            pick the top_k of them as recommendation
        '''
        iids.sort_values(ascending=False, inplace=True)
        G_recommendation = list(iids.index)[:top_k]
        # print("Top-{} items recommended for the Group".format(top_k))
        # print(G_recommendation)
        return G_recommendation



    def agg_G_Popular(self, top_k, dataset, iids_G_common_not_rated ):
        '''
            Popularity Aggreagation Strategy
        '''
        iids = dataset.groupby('iid')['rating'].count()
        iids = iids.loc[list(iids_G_common_not_rated)]
        iids.sort_values(ascending=False, inplace=True)
        return self.pick_top_k_recommendations(iids, top_k)

    def agg_G_avg(self, top_k, G_predicted_df):
        '''
            Average Aggregation Strategy
        '''
        iids = G_predicted_df.mean(axis=0)
        return self.pick_top_k_recommendations(iids, top_k)

    def agg_G_LM(self, top_k, G_predicted_df):
        '''
            Least Misery Aggregation Strategy
        '''
        iids = G_predicted_df.min()
        return self.pick_top_k_recommendations(iids, top_k)

    def agg_G_MP(self, top_k, G_predicted_df):
        '''
            Most Pleasure Aggregation Strategy
        '''
        iids = G_predicted_df.max()
        return self.pick_top_k_recommendations(iids, top_k)

    def agg_G_Mult(self, top_k, G_predicted_df):
        '''
            Multiplicative Aggregation Strategy
        '''
        iids = G_predicted_df.apply(lambda x: np.prod(x))
        return self.pick_top_k_recommendations(iids, top_k)

    def agg_G_Borda(self, top_k, G_predicted_df):
        '''
            Borda Count Aggregation Strategy
        '''
        # is rank per user
        rank_df = G_predicted_df.transpose().rank()
        iids = rank_df.apply(lambda x: np.sum(x), axis=1)
        return self.pick_top_k_recommendations(iids, top_k)

    def agg_G_Approval(self, top_k, G_predicted_df):
        '''
            Approval voting
        '''
        threshold = 3.0
        iids = G_predicted_df.apply(lambda x: x > threshold).sum()
        return self.pick_top_k_recommendations(iids, top_k)

    def agg_G_MRP(self, top_k, G_members_has_rated_number, G_predicted_df, G):
        '''
            Most Respected Person Aggregation Strategy
        '''
        # get the number of items rated by each user
        most_respected = -1
        max_items_rated = 0
        for member in G:
            num_rated_items = G_members_has_rated_number[member]
            # print(member)
            # print(num_rated_items)
            if (num_rated_items > max_items_rated):
                max_items_rated = num_rated_items
                most_respected = member

        print("The 'expert' or 'most respected' user is: {}".format(most_respected))
        iids = G_predicted_df.loc[most_respected]
        return self.pick_top_k_recommendations(iids, top_k)

    def agg_G_AWM(self, top_k, G_predicted_df):
        '''
            Average Without Misery Aggregation Strategy
        '''
        threshold = 2
        G_AWM = G_predicted_df.apply(lambda col: col > threshold)
        # Replace False with NaN
        G_AWM.replace(False, np.nan, inplace=True)
        # print(G_AWM)
        # print(G_AWM[2064].isnull().sum())
        # Drop the columns with any element being NaN
        G_AWM.dropna(axis=1, how='any', inplace=True)
        # Get the column names
        G_AWM_columns = list(G_AWM.columns)
        # Get a DF with the reamining columns
        G_AWM = G_predicted_df[G_AWM_columns]
        # Get the average
        iids = G_AWM.mean(axis=0)
        return self.pick_top_k_recommendations(iids, top_k)

    def agg_G_Add(self, top_k, G_predicted_df):
        '''
            Additive Aggregation Strategy
        '''
        iids = G_predicted_df.apply(lambda x: np.sum(x))
        return self.pick_top_k_recommendations(iids, top_k)

    def predict_for_G(self, agg_model, top_k):
        return agg_model(top_k)