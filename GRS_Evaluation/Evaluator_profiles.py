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
        # _user_predicted_df = argv[0]
        # cos_sim_matrix = arg[1]
        # _user_recommendations = arg[2]
        # model = arg[3]
        # groups = arg[4]
        # v_users = arg[5]

        print("@@@ get evaluations @@@")
        print(metric)
        user_predicted_df = argv[0]
        ##print(user_predicted_df)
        item_cos_sim_matrix_df = argv[1]
        ##print(item_cos_sim_matrix_df)
        user_recommendations = argv[2]
        ##print(user_recommendations)
        model = argv[3]
        ##print(model)
        Gs = argv[4]
        ##print(Gs)
        v_users = argv[5]
        ##print(v_users)
        iids = argv[6]

        function_G_results = []
        for i, G in enumerate(Gs):
            print("Calculating {} for virtual user {} formed by {}".format(metric, v_users[i], Gs[i]))

            if metric == 'hr':
                hit_rate = self.get_hit_rate(v_users[i], user_predicted_df, user_recommendations,
                                             p.NUM_ITEMS_TO_RECOMMEND, Gs[i], model)#G_member_predictions_list[i], model)
                print("{} for this group: {}".format(metric, hit_rate))
                #self.append_results(agg_function, hit_rate)
                function_G_results.append(hit_rate)

            if metric == "ndcg":
                ndcg = self.get_ndcg(v_users[i], user_predicted_df, user_recommendations,
                                             p.NUM_ITEMS_TO_RECOMMEND, Gs[i], model, p.at_k) #, G_member_predictions_list[i], model)
                #ndcg = self.get_ndcg_members_avg(G_recommendations_list[i], G_predicted_dfs[i], p.at_k, G) #, G_member_predictions_list[i], model)

                function_G_results.append(ndcg)

            if metric == "diversity":
                # Calculate intralist similarity (ILS)
                diversity = self.get_diversity(v_users[i], user_predicted_df, user_recommendations,
                                               Gs[i], item_cos_sim_matrix_df, model)

                function_G_results.append(diversity)

            if metric == "coverage":
                coverage = self.get_coverage_group(v_users[i], user_predicted_df, user_recommendations,
                                             p.NUM_ITEMS_TO_RECOMMEND, Gs[i], model)
                ##coverage = self.get_v_user_predictions(v_users[i], user_recommendations)
                function_G_results.append(coverage)


        print("Function G results")
        print(function_G_results)

        self._metric_results[metric].append(function_G_results)
        #total_avg = np.mean(function_G_results)

        ##if metric == 'coverage':
        ##    total_avg = self.get_coverage_catalog(iids, function_G_results)
        ##else:
        total_avg = np.median(function_G_results)

        print("Total {} for all {} groups is: {}".format(metric, len(Gs), total_avg))
        self._avg_metric_results[metric] = total_avg

        ## print("exiting in Evaluator_profiles.get_evaluations()")
        ## sys.exit(3)


    def get_hit_rate(self, v_user, user_predicted_df, user_recommendations, top_k, G, model): #G_member_predictions_list, model):
        v_user_recommendations = set(user_recommendations[v_user])
        print(v_user_recommendations)
        print()

        G_hit_rates = []

        for member in G:
            member_ind_recommend = set(user_recommendations[member])
            hits = len(member_ind_recommend.intersection(v_user_recommendations))
            ##print(hits)

            member_hit_rate = hits / top_k
            #print("member:",member, "hit rate:",member_hit_rate)
            G_hit_rates.append(member_hit_rate)

        print("median:",np.median(G_hit_rates))
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

    def get_coverage_group(self, v_user, user_predicted_df, user_recommendations, top_k, G, model):
        v_user_recommendations = set(user_recommendations[v_user])
        print(v_user_recommendations)
        print()

        G_all_recommended_items = []
        for member in G:
            member_ind_recommend = user_recommendations[member]
            G_all_recommended_items = G_all_recommended_items + member_ind_recommend

        #print(len(G_all_recommended_items))
        N = set(G_all_recommended_items)
        ##print("N: {}".format(N))
        ##print(len(N))

        n = v_user_recommendations
        ##print("n: {}".format(n))
        ##print(len(n))

        n = n.intersection(N)
        ##print("n intersect: {}".format(n))
        ##print(len(n))

        coverage = len(n) / len(N)
        ##print("Coverage: {}".format(coverage))
        return coverage

    def get_v_user_predictions(self, v_user, user_recommendations):
        """

        """
        return set(user_recommendations[v_user])


    def get_coverage_catalog(self, iids, all_v_user_recommendations):
        """

        """
        n = list(set(chain.from_iterable(all_v_user_recommendations)))
        #print("n")
        #print(n)
        n = len(n)
        #print(n)
        N = len(iids)
        #print(N)
        coverage = n / N

        return coverage



    def get_ndcg(self, v_user, user_predicted_df, user_recommendations, top_k, G, model, at_k): #G_member_predictions_list, model):
        v_user_recommendations = user_recommendations[v_user]
        #print("v user recommendations")
        #print(v_user_recommendations)
        #print()

        G_ndcg = []
        for member in G:
            member_ind_recommend = user_predicted_df.loc[member]
            member_ind_recommend = member_ind_recommend[v_user_recommendations]
            member_ind_recommend = member_ind_recommend.dropna()
            member_ndcg = ndcg_at_k(member_ind_recommend, at_k)
            #print("nDCG for this member {} is: {}".format(member, member_ndcg))
            G_ndcg.append(member_ndcg)

        print(G_ndcg)
        ##print(np.median(G_ndcg))

        return np.median(G_ndcg)


    def get_ndcg_members_avg(self, G_recommendations, G_predicted_df, at_k, G):
        G_ndcg = []
        print("Recommendations for this group:")
        print(G_recommendations)
        ##print(len(G_recommendations))

        print("Top {} recommendations according to 'at_k'".format(at_k))
        G_recommendations_at_k = G_recommendations[:at_k]
        print(G_recommendations_at_k)

        print(G_predicted_df)
        G_predicted_df = G_predicted_df[G_recommendations_at_k]
        print(G_predicted_df)
        print("=" * 50)

        for member in G:
            member_ind_recommend = list(G_predicted_df.loc[member])
            print(member_ind_recommend)
            member_ndcg = ndcg_at_k(member_ind_recommend, at_k)
            print("nDCG for this member {} is: {}".format(member, member_ndcg))
            G_ndcg.append(member_ndcg)

        print(np.median(G_ndcg))
        sys.exit(3)
        return np.median(G_ndcg)





    def get_diversity(self, v_user, user_predicted_df, user_recommendations, G, item_cosine_sim_matrix_df, model):
        v_user_recommendations = user_recommendations[v_user]
        #print("v_user_recommendations")
        print(v_user_recommendations)
        print()
        #ILS
        cos_sim_df = item_cosine_sim_matrix_df.loc[v_user_recommendations][v_user_recommendations]
        upper_right = np.triu_indices(cos_sim_df.shape[0],  k=1)
        diversity = np.mean(np.array(cos_sim_df)[upper_right])
        diversity = 1 - diversity # diversity = 1 - ILS

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
    idcg = dcg_at_k( sorted( r, reverse=True ), k) # , method=0)

    if not idcg:
        return 0.

    if idcg < 0 :
        print("~~~ NAN ~~~")
        return 0.

    #return dcg_at_k_(r, k, method=0) / idcg
    #print("idcg:",idcg)

    return dcg_at_k(r, k) / idcg

##########################
def dcg_at_k_(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(3, r.size + 2))) ### fix here
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.




###################

if __name__ == "__main__":
    ev = Evaluator()
    agg_function = 'AVG'
    G_recommendations = 2
    G_predicted_df = 3
    top_k = 5

    ev.get_evaluations(agg_function, G_recommendations, G_predicted_df, top_k)