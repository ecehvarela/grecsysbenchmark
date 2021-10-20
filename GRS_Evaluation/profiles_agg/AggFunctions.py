'''
Class having the different rating aggregation functions

'''
import sys
import numpy as np
import pandas as pd
import GRS_Evaluation.profiles_agg.Parameters as p


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


    def most_popular_item(self, dataset):
        items_count = dataset.groupby('iid')['uid'].agg('count').sort_values(ascending=False)
        #print("ITEMS count")
        #print(items_count)
        most_popular = list(items_count.head(5).index)[0]
        #print("Most popular item:", most_popular)

        # Find mean
        means = dataset.groupby('iid')['rating'].mean()
        mean_rating = means[most_popular]
        #print("rating of most popular item:", mean_rating)
        # Create a small Series
        iid = {most_popular:mean_rating}
        #print(iid)
        iid = pd.Series(iid)
        print(iid)

        #print("exiting in AggFunctions.most_popular_item()")
        #sys.exit(3)
        return iid

###
    def agg_G_Popular_2(self, G_predicted_df, dataset ):
        '''
            Popularity Aggreagation Strategy
        '''
        ids_items = list(G_predicted_df.columns)
        # from the original _dataset, count the ratings for this items
        count_ratings = dataset.groupby('iid')['rating'].count()
        # get only the ones from G
        count_ratings = count_ratings[ids_items]
        # get only those with more than X ratings
        iids = list(count_ratings[lambda x: x > p.min_pop])

        # Get the average of these popular iids
        means = dataset.groupby('iid')['rating'].mean()
        iids = means[iids]

        #print("len iids:",len(iids))
        #print(type(iids))
        #print(iids)

        if len(iids) == 0:
            print("This virtual user does not have a profile at this point")
            print("getting the most popular item")
            # Find the most popular item and get its average rating
            iids = self.most_popular_item(dataset)

        print(len(iids))
        return iids
        #return self.pick_top_k_recommendations(iids, top_k)
###


    def agg_G_Popular(self, G_predicted_df, dataset ):
        '''
            Popularity Aggreagation Strategy
        '''
        ids_items = list(G_predicted_df.columns)
        # from the original _dataset, count the ratings for this items
        count_ratings = dataset.groupby('iid')['rating'].count()
        # get only the ones from G
        count_ratings = count_ratings[ids_items]
        # get only those with more than X ratings
        iids = list(count_ratings[lambda x: x > p.min_pop])

        # Get the average of these popular iids
        means = dataset.groupby('iid')['rating'].mean()
        iids = means[iids]

        #print("len iids:",len(iids))
        #print(type(iids))
        #print(iids)

        if len(iids) == 0:
            print("This virtual user does not have a profile at this point")
            print("getting the most popular item")
            # Find the most popular item and get its average rating
            iids = self.most_popular_item(dataset)

        return iids
        #return self.pick_top_k_recommendations(iids, top_k)

    def agg_G_avg(self, G_predicted_df):
        '''
            Average Aggregation Strategy
        '''
        G_predicted_df = G_predicted_df.replace(0, np.nan)
        iids = G_predicted_df.mean(axis=0, skipna=True)
        return iids
        #return self.pick_top_k_recommendations(iids, top_k)

    def agg_G_LM(self, G_predicted_df):
        '''
            Least Misery Aggregation Strategy
        '''
        G_predicted_df = G_predicted_df.replace(0, np.nan)
        iids = G_predicted_df.min()
        ##print(iids.shape)
        return iids
        #return self.pick_top_k_recommendations(iids, top_k)

    def agg_G_MP(self, G_predicted_df):
        '''
            Most Pleasure Aggregation Strategy
        '''
        G_predicted_df = G_predicted_df.replace(0, np.nan)
        iids = G_predicted_df.max()
        ##print(iids.shape)
        return iids
        #return self.pick_top_k_recommendations(iids, top_k)

    def agg_G_Mult(self, G_predicted_df):
        '''
            Multiplicative Aggregation Strategy
        '''
        G_predicted_df = G_predicted_df.replace(0, np.nan)
        iids = G_predicted_df.apply(lambda x: np.prod(x))

        #print(iids)
        upper = iids.max()
        lower = iids.min()
        range_upper = 5.0
        range_lower = 1.0

        #if lower < range_upper:
        #    range_lower = lower

        oldRange = upper - lower
        newRange = range_upper - range_lower

        print("max: {}, min: {} to max:{}, min: {}".format(upper, lower, range_upper, range_lower))
        #print(iids.sort_values(ascending=False))
        if upper > range_upper:
            print("Adjusting ratings")
            iids = iids.apply(lambda x: ( ( (x - lower) * newRange) / oldRange) + range_lower)
        #print("l_norm")
        #print(iids)

        return iids
        #return self.pick_top_k_recommendations(iids, top_k)


    def agg_G_Borda(self, G_predicted_df, dataset):
        '''
            Borda Count Aggregation Strategy
        '''
        #print(G_predicted_df)
        # is rank per user
        rank_df = G_predicted_df.transpose().rank()
        #print(rank_df)
        iids = rank_df.apply(lambda x: np.sum(x), axis=1)
        #print(iids)
        iids =  self.pick_top_k_recommendations(iids, p.top_bc)

        #print(iids)
        # Get the average of these popular iids
        means = dataset.groupby('iid')['rating'].mean()
        iids = means[iids]

        #print(iids)
        return iids

    def agg_G_Approval(self, G_predicted_df, dataset):
        '''
            Approval voting
        '''
        threshold = 3.0

        G_predicted_df[G_predicted_df <= threshold] = 0
        G_predicted_df = G_predicted_df.replace(0, np.nan)
        G_predicted_df.dropna(axis=1, how='all', inplace=True)
        iids = G_predicted_df.mean(axis=0, skipna=True)

        #print(iids)
        return iids
        #return self.pick_top_k_recommendations(iids, top_k)

    def agg_G_MRP(self, G_predicted_df, dataset):
        '''
            Most Respected Person Aggregation Strategy
        '''
        #print(G_predicted_df)
        # get the number of items rated by each user
        most_respected = -1
        max_items_rated = 0
        G = list(G_predicted_df.index)
        #print(G)
        G_predicted_df = G_predicted_df.replace(0, np.nan)
        for member in G:
            num_rated_items = G_predicted_df.loc[member].count()
            print("member: {}, rated items:{}".format(member, num_rated_items))
            if (num_rated_items > max_items_rated):
                max_items_rated = num_rated_items
                most_respected = member

        print("The 'expert' or 'most respected' user is: {}".format(most_respected))
        iids = G_predicted_df.loc[most_respected]
        iids.dropna(inplace=True)
        #print(iids)
        return iids
        #return self.pick_top_k_recommendations(iids, top_k)


    def agg_G_AWM(self, G_predicted_df):
        '''
            Average Without Misery Aggregation Strategy
        '''
        threshold = 2
        #print(G_predicted_df)
        G_AWM = G_predicted_df.apply(lambda col: col > threshold)
        #print(G_AWM)
        # Replace False with NaN
        G_AWM.replace(False, np.nan, inplace=True)
        # Drop the columns with all element being NaN
        G_AWM.dropna(axis=1, how='all', inplace=True)
        # Get the column names
        G_AWM_columns = list(G_AWM.columns)
        # Get a DF with the reamining columns
        G_AWM = G_predicted_df[G_AWM_columns]
        iids = G_AWM.replace(0.0, np.nan)
        # Get the average
        iids = iids.mean(axis=0, skipna=True)
        #print(iids)

        return iids
        #return self.pick_top_k_recommendations(iids, top_k)

    def agg_G_Add(self, G_predicted_df):
        '''
            Additive Aggregation Strategy
        '''
        G_predicted_df = G_predicted_df.replace(0, np.nan)
        iids = G_predicted_df.apply(lambda x: np.sum(x))
        #print(iids)
        upper = iids.max()
        lower = iids.min()
        range_upper = 5.0
        range_lower = 1.0

        #if lower < range_upper:
        #    range_lower = lower

        oldRange = upper - lower
        newRange = range_upper - range_lower

        print("max: {}, min: {} to max:{}, min: {}".format(upper, lower, range_upper, range_lower))
        # print(iids.sort_values(ascending=False))

        if upper > range_upper:
            print("Adjusting ratings")
            iids = iids.apply(lambda x: (((x - lower) * newRange) / oldRange) + range_lower)

        # print("l_norm")
        #print(iids)
        return iids

        #return self.pick_top_k_recommendations(iids, top_k)

    def predict_for_G(self, agg_model, top_k):
        return agg_model(top_k)

'''
        G_predicted_df = G_predicted_df.replace(0, np.nan)
        iids = G_predicted_df.apply(lambda x: np.prod(x))

        #print(iids)
        upper = iids.max()
        lower = iids.min()
        range_upper = 5.0
        range_lower = 1.0
        oldRange = upper - lower
        newRange = range_upper - range_lower

        print("max: {}, min: {} to max:{}, min: {}".format(upper, lower, range_upper, range_lower))
        #print(iids.sort_values(ascending=False))
        iids = iids.apply(lambda x: ( ( (x - lower) * newRange) / oldRange) + range_lower)
        #print("l_norm")
        #print(iids)

'''