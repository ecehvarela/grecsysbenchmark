
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
import GRS_Evaluation.ratings_agg.Parameters as p

def load_dataset(ratings_filename):
    data = pd.read_csv(ratings_filename)
    print(data.head())
    return data


def get_pivot_table(data):
    #print("MMMMMMMMMMM")
    #print(data)
    #print(type(data))
    data = pd.pivot(data, index='uid', columns='iid', values='rating')
    data.fillna(0, inplace=True)
    #print("[INFO] data loaded from {}".format(ratings_filename))
    return data

def form_clusters(X, k=2):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)

    return kmeans.labels_

def form_similar_group(dataset, seed_user):
    print("Group is formed by Similar users")
    # get the cluster for the seed user
    user_cluster = dataset.loc[seed_user]['cluster']
    print("User from cluster: {}".format(user_cluster))
    print("find similar users from same cluster")
    # get only the users in that cluster
    dataset = dataset[dataset['cluster'] == user_cluster]
    # get the user-user similarity matrix
    X = dataset.drop('cluster', axis=1)
    X_indices = list(X.index)
    sim_matrix = cosine_similarity(X)
    # Create a DF
    sim_df = pd.DataFrame(sim_matrix, index=X_indices, columns=X_indices)
    user_sims = sim_df.loc[seed_user]
    user_sims.sort_values(ascending=False, inplace=True)
    user_sims_list = list(user_sims.index) #[1:]

    return user_sims_list


def form_dissimilar_group(dataset, seed_user):
    print("Group is formed by Dissimilar users")
    # get the cluster for the seed user
    user_cluster = dataset.loc[seed_user]['cluster']
    print("User from cluster: {}".format(user_cluster))
    print("find most dissimilar users from the other cluster")
    # Get the seed user vector
    seed_user_series = dataset.loc[seed_user]
    # get the users from the other cluster
    dataset = dataset[dataset['cluster'] != user_cluster]
    dataset = dataset.append(seed_user_series, ignore_index=False)
    # get the user-user similarity matrix
    X = dataset.drop('cluster', axis=1)
    X_indices = list(X.index)
    sim_matrix = cosine_similarity(X)
    # Create a DF
    sim_df = pd.DataFrame(sim_matrix, index=X_indices, columns=X_indices)
    user_sims = sim_df.loc[seed_user]
    user_sims.sort_values(ascending=False, inplace=True)
    user_sims_list = list(user_sims.index)  # [1:]

    return user_sims_list


def form_real_group(dataset, seed_user, G_size):
    print("Group is formed by similar and dissimilar users")

    user_sims_list = form_similar_group(dataset, seed_user)
    G_sim = set(user_sims_list[:G_size])
    user_sims_list = form_dissimilar_group(dataset, seed_user)
    G_dis = set(user_sims_list[-G_size + 1:])
    # join to sets
    sim_dis = G_sim.union(G_dis)
    # remove seed from this set
    sim_dis.remove(seed_user)
    # randomly pick G_size-1 elements
    print("Randomly pick similar and dissimilar users")
    random_members = np.random.choice(list(sim_dis), G_size-1, replace=False)
    G_real = {seed_user}
    G_real = G_real.union(set(random_members))

    return G_real


def get_group(dataset, G_type, G_size):
    # Get a random user from the uid
    random_user = np.random.choice(list(dataset.index))
    print("Seed user: {}".format(random_user))

    if G_type == 's':
        # Similar users
        user_sims_list = form_similar_group(dataset, random_user)
        G = set(user_sims_list[:G_size])

    elif G_type == 'd':
        # Dissimilar users
        user_sims_list = form_dissimilar_group(dataset, random_user)
        G = set(user_sims_list[-G_size+1:])
        G.add(random_user)


    elif G_type == 'rl':
        # Real users
        G = form_real_group(dataset, random_user, G_size)

    else:
        # Random users
        pass

    return np.array(list(G))



if __name__ == "__main__":
    G_type = 'd'
    G_size = 4
    dataset = 'op'
    ratings_filename = p.data_path + "/" + dataset.upper() + "/" + "ratings.csv"

    # Load the dataset and create a UxI matrix
    data = load_dataset(ratings_filename)
    data = get_pivot_table(data)
    print("Total users: {}".format(data.shape[0]))
    # Create 2 clusters
    cluster_labels = form_clusters(data)
    print('Total labels: {}'.format(len(cluster_labels)))
    data['cluster'] = cluster_labels
    #print(data.head())
    print()
    G = get_group(data, G_type, G_size)
    print("Group: {}".format(G))






