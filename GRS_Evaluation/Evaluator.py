'''

'''

import numpy as np
import sys
from collections import defaultdict
from statistics import mean
import itertools
import pandas as pd
from itertools import chain

import GRS_Evaluation.ratings_agg.Parameters as p


class Evaluator():

    def __init__(self):
        self._results = {}
        self._group_results = defaultdict(list)
        self._metric_results = defaultdict(list)
        self._avg_metric_results = {}
        self._group_avgs = {}

    def get_results(self):
        '''

        '''
        return self._results

    def get_avg_group_results(self):
        '''

        '''
        for k, result in self._group_results.items():
            res_avg = mean(result)
            self._group_avgs[k] = res_avg

    def get_metric_results(self, metric):
        """

        :param metric:
        :return:
        """
        return self._metric_results[metric]


    def append_group_results(self):
        '''

        '''
        for k, result in self._results.items():
            self._group_results[k].append(result)


    def view_results(self):
        '''

        '''
        for k, result in self._results.items():
            print("Results for {}: {}".format(k, result))


    def view_group_results(self):
        '''

        '''
        for k, result in self._group_avgs.items():
            print("Group results for {}: {}".format(k, result))

    def get_group_results(self):
        '''

        '''
        return self._group_avgs


    def append_results(self, agg_function, result):
        '''

        '''
        self._results[agg_function] = result


    def get_evaluations(self, metric, *argv):
        #print("First argument :", agg_function)
        # G_recommendations = argv[0]
        # G_predicted_df = argv[1]
        # G = argv[]
        # cos_sim_matrix = arg[2]
        # model = arg[3
        #### grs.G_member_predictions_list = arg[4]
        # grs._iids = arg[4]

        print("@@@ get evaluations @@@")
        print(metric)
        G_recommendations_list = argv[0][0]
        print(G_recommendations_list)
        G_predicted_dfs = argv[1]
        print(G_predicted_dfs)
        Gs = [list(group.index) for group in argv[1]]
        print(Gs)
        item_cosine_sim_matrix_df = argv[2]
        model = argv[3]
        print(model)
        ###G_member_predictions_list = argv[4]
        iids = argv[4]


        function_G_results = []
        for i,G in enumerate(Gs):
            print("Calculating {} for group {} formed by {}".format(metric, i+1, G))

            if metric == 'hr':
                hit_rate = self.get_hit_rate(G_recommendations_list[i], G_predicted_dfs[i],
                                             p.NUM_ITEMS_TO_RECOMMEND, G, model)#G_member_predictions_list[i], model)
                print("{} for this group: {}".format(metric, hit_rate))
                #self.append_results(agg_function, hit_rate)
                function_G_results.append(hit_rate)

            if metric == "ndcg":
                ndcg = self.get_ndcg(G_recommendations_list[i], G_predicted_dfs[i],
                                     p.at_k, G) #, G_member_predictions_list[i], model)
                function_G_results.append(ndcg)

            if metric == "diversity":
                # Calculate intralist similarity (ILS)
                diversity = self.get_diversity(G_recommendations_list[i], G_predicted_dfs[i], G, item_cosine_sim_matrix_df)

                function_G_results.append(diversity)

            if metric == "coverage":
                coverage = self.get_coverage_group(G_recommendations_list[i], G_predicted_dfs[i], G)
                ##coverage = self.get_G_predictions(G_recommendations_list[i], G_predicted_dfs[i], G)
                function_G_results.append(coverage)


        print(function_G_results)
        self._metric_results[metric].append(function_G_results)

        ##if metric == 'coverage':
        ##    total_avg = self.get_coverage_catalog(iids, function_G_results)
        ##else:
        #total_avg = np.mean(function_G_results)
        total_avg = np.median(function_G_results)


        print("Total {} for all {} groups is: {}".format(metric, len(Gs), total_avg))
        self._avg_metric_results[metric] = total_avg


        #print("exiting at Evaluator.get_evaluations()")
        #sys.exit(4)

    def get_hit_rate(self, G_recommendations, G_predicted_df, top_k, G, model): #G_member_predictions_list, model):
        G_recommendations = set(G_recommendations)
        G_hit_rates = []

        if model == 'xxx':
            pass
            '''
            for member in G:
                member_ind_recommend = set(list(G_member_predictions_list[member].index)[:top_k])
                ##print(member_ind_recommend)
                ##print(len(member_ind_recommend))

                hits = len(member_ind_recommend.intersection(G_recommendations))
                member_hit_rate = hits / top_k
                ##print("member:",member, "hit rate:",member_hit_rate)
                G_hit_rates.append(member_hit_rate)
            '''

        else:
            for member in G:
                member_ind_recommend = G_predicted_df.loc[member]
                ##member_ind_recommend.sort_values(ascending=False, inplace=True)
                member_ind_recommend = member_ind_recommend.reset_index(drop=False)
                #print(member_ind_recommend.columns)
                member_ind_recommend.sort_values([member,'index'], inplace=True, ascending=False)

                ##print("Member individual recommendations")
                member_ind_recommend = set(member_ind_recommend['index'][:top_k])
                ##print(member_ind_recommend)
                ##print(G_recommendations)

                hits = len(member_ind_recommend.intersection(G_recommendations))
                ##print(hits)

                member_hit_rate = hits / top_k
                ##print("member:",member, "hit rate:",member_hit_rate)
                G_hit_rates.append(member_hit_rate)

        ##print(np.median(G_hit_rates))

        return np.median(G_hit_rates)
        #return np.mean(G_hit_rates)


    def get_coverage(self, G_recommendations, G_predicted_df, G):
        G_coverage = []
        for member in G:
            # get all of user's predicted ratings
            member_ind_recommend = G_predicted_df.loc[member]
            # sort and pick the top-k recommendations
            member_ind_recommend = member_ind_recommend.sort_values(ascending=False)[:p.NUM_ITEMS_TO_RECOMMEND]
            member_ind_recommend_set = set(member_ind_recommend.index)
            # Check how many of the G_recommendations are in the member_ind_recommend
            match_items = len(member_ind_recommend_set.intersection(set(G_recommendations)))
            # Calculate the coverage
            member_coverage = match_items / len(G_recommendations)
            G_coverage.append(member_coverage)

        ##print(np.median(G_coverage))
        return np.median(G_coverage)

    def get_coverage_group(self, G_recommendations, G_predicted_df, G):
        G_all_recommended_items = []
        for member in G:
            member_ind_recommend = G_predicted_df.loc[member]
            ##print("Member {} recommendation".format(member))
            # sort and pick the top-k recommendations
            member_ind_recommend = member_ind_recommend.sort_values(ascending=False)[:p.NUM_ITEMS_TO_RECOMMEND]
            member_ind_recommend = list(member_ind_recommend.index)
            ##print(member_ind_recommend)
            G_all_recommended_items = G_all_recommended_items + member_ind_recommend

        #print(len(G_all_recommended_items))
        N = set(G_all_recommended_items)
        ##print("N: {}".format(N))
        ##print(len(N))

        n = set(G_recommendations)
        ##print("n: {}".format(n))
        ##print(len(n))

        n = n.intersection(N)
        ##print("n intersect: {}".format(n))
        ##print(len(n))

        coverage = len(n) / len(N)
        ##print("Coverage: {}".format(coverage))
        return coverage

    def get_G_predictions(self, G_recommendations, G_predicted_df, G):
        """

        """
        return set(G_recommendations)

    def get_coverage_catalog(self, iids, all_G_recommendations):
        """

        """
        n = list(set(chain.from_iterable(all_G_recommendations)))
        #print("n")
        #print(n)
        n = len(n)
        #print(n)
        N = len(iids)
        #print(N)
        coverage = n / N

        return coverage



    def get_ndcg(self, G_recommendations, G_predicted_df, at_k, G): #, G_member_predictions_list, model):
        #print(model)
        G_ndcg = []
        ##print("Recommendations for this group:")
        ##print(G_recommendations)
        ##print(len(G_recommendations))
        ##print("Top {} recommendations according to 'at_k'".format(at_k))
        G_recommendations_at_k = G_recommendations[:at_k]
        ##print(G_recommendations_at_k)
        G_predicted_df = G_predicted_df[G_recommendations_at_k]
        #print(G_predicted_df)
        #print("=" * 50)

        for member in G:
            member_ind_recommend = list(G_predicted_df.loc[member])
            member_ndcg = ndcg_at_k(member_ind_recommend, at_k)
            ##print("nDCG for this member {} is: {}".format(member, member_ndcg))
            G_ndcg.append(member_ndcg)

        ##print(np.median(G_ndcg))
        return np.median(G_ndcg)

    def get_diversity(self, G_recommendations, G_predicted_df, G, item_cosine_sim_matrix_df):
        #ILS
        cos_sim_df = item_cosine_sim_matrix_df.loc[G_recommendations][G_recommendations]
        upper_right = np.triu_indices(cos_sim_df.shape[0],  k=1)

        diversity = np.mean(np.array(cos_sim_df)[upper_right])
        diversity = 1 - diversity       # diversity = 1 - ILS

        #print(diversity)
        return diversity


    def get_avg_metric_result(self, metric):
        """

        """
        return self._avg_metric_results[metric]


    def save_group_results_csv(self, group_size, group_type,agg_strategy,
                               agg_function,dataset,metric):
        '''
            group_size'_'group_type'_'agg_strategy'_'agg_function'_'dataset'_'metric'.csv
        '''


        header = ",".join([k for k, v in self._group_avgs.items()])
        results = ",".join([str(v) for k, v in self._group_avgs.items()])

        print(header)
        print(results)

        # Files output name will be formed as
        #  'group_size'_'group_type'_'agg_strategy'_'agg_function'_'dataset'_'metric'.csv


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum( np.subtract( np.power( 2, r ), 1) / np.log2( np.arange( 2, r.size + 2 ) ) )

def ndcg_at_k(r, k):
    idcg = dcg_at_k( sorted( r, reverse=True ), k )
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg


###################

if __name__ == "__main__":
    ev = Evaluator()
    agg_function = 'AVG'
    G_recommendations = 2
    G_predicted_df = 3
    top_k = 5

    ev.get_evaluations(agg_function, G_recommendations, G_predicted_df, top_k)