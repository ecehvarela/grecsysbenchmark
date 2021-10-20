#!/usr/bin/env python
# coding: utf-8
'''
    For each item aggregate all of its reviews
    then, using Rake, extract all of its keyworkds

    It creates the cosine-similarity, and the tfidf matrix

'''
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sys
import re
import pandas as pd
import gzip
import os
import pickle
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def get_output_path(filename):
    path = filename.split('/')[:-1]
    path = "/".join(path)
    return path + "/"


def save_object(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)
        print("Object was saved as {}".format(filename))

def load_object(filename, in_filename):
    input_path = get_output_path(in_filename)
    object_filename = input_path + filename
    #print(object_filename)

    file = open(object_filename, 'rb')
    data = pickle.load(file)
    file.close()

    return data


def load_object_ML(filename):
    object_filename =  filename
    print(object_filename)

    file = open(object_filename, 'rb')
    data = pickle.load(file)
    file.close()

    return data


def save_df_to_csv(df, filename):
    df.to_csv(filename, header=True, index=False)
    print("Ratings file was saved as {}".format(filename))


def check_out_filename(out_filename, in_filename):
    if out_filename == None:
        input_path = get_output_path(in_filename)
        return input_path + "matrices.obj"

    else:
        return out_filename


def extract_categories(row):
    categories = []
    for category in row[0]:
        category = category.lower().replace(' ', '_')
        category = re.sub('[^a-zA-Z0-9 \n\.]', '', category)
        categories.append(category)

    categories = ",".join([c for c in categories])
    return categories

def load_metadata(in_filename):
    # asin, categories

    df = getDF(in_filename)
    df = df[['asin', 'categories']].rename(columns={"asin": "iid"})

    # map asin to iid
    iid2id = load_object("iid2id.dict", in_filename)
    df['iid'] = df['iid'].map(iid2id)

    df.dropna(how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df.astype({'iid': int})
    df['categories'] = df['categories'].apply(extract_categories)
    df.sort_values(by=['iid'], ascending=True, inplace=True)
    df.set_index('iid', inplace=True)
    #df_indices = list(df.index)

    return df


def extract_tfidf_df(df):
    df_indices = list(df.index)
    # Get the tf-idf matrix
    tfdif = TfidfVectorizer()
    tfidf_matrix = tfdif.fit_transform(df['categories']).todense()
    # print(tfidf_matrix)
    # Convert it to DF
    tfid_matrix_df = pd.DataFrame(tfidf_matrix, index=df_indices)
    # print(tfid_matrix_df)
    return tfid_matrix_df, df_indices

def extract_cosine_matrix_df(tfidf_matrix, df_indices):
    # Get rhe cosine similarity matrix
    ##cosine_sim_matrix = cosine_similarity(count_matrix, count_matrix)
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # Convert if as DF
    cosine_sim_matrix_df = pd.DataFrame(cosine_sim_matrix,
                                        index=df_indices,
                                        columns=df_indices)
    # print(cosine_sim_matrix_df)
    return cosine_sim_matrix_df


def get_matrices(filename, out_filename, save=0):
    print("filename:", filename)
    df = load_metadata(filename)
    tfid_matrix_df, indices = extract_tfidf_df(df)
    cosine_sim_matrix_df = extract_cosine_matrix_df(tfid_matrix_df, indices)

    data = [tfid_matrix_df, cosine_sim_matrix_df]
    if save:
        save_object(data, out_filename)
    else:
        return data


########################################
if __name__ == "__main__":
    # Get the parameters
    # construct the argument parse and parse the arguments
    os.chdir(os.pardir)
    currentDirectory = os.getcwd()
    print("Current working directory:", currentDirectory)

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input file location")
    ap.add_argument("-o", "--output", required=False, help="Output file location")

    args = vars(ap.parse_args())

    # os.chdir(os.pardir)
    # currentDirectory = os.getcwd()
    # data_path = 'data/'
    # in_filename = 'data/AUTO/reviews_Automotive_5.json.gz'
    in_filename = currentDirectory + "/" +args["input"]
    out_filename = check_out_filename(args["output"], in_filename)
    print("input file:{}, output file:{}".format(in_filename, out_filename))


    data = get_matrices(in_filename, out_filename)

    ### This is extra for the test ##
    tfidf_matrix_df = data[0]
    cosine_sim_matrix_df = data[1]

    print("tfidf")
    print(tfidf_matrix_df)
    print("cosine sim")
    print(cosine_sim_matrix_df)

    sys.exit(9)

    # asin, categories
    df = getDF(in_filename)
    df = df[['asin', 'categories']].rename(columns={"asin": "iid"})

    # map asin to iid
    iid2id = load_object("iid2id.dict")
    df['iid'] = df['iid'].map(iid2id)

    df.dropna(how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df.astype({'iid': int})
    df['categories'] = df['categories'].apply(extract_categories)
    df.sort_values(by=['iid'], ascending=True, inplace=True)
    df.set_index('iid', inplace=True)

    df_indices = list(df.index)
    #print(df_indices)
    # Get the count matrix
    ##count = CountVectorizer()
    ##count_matrix = count.fit_transform(df['categories']).todense()
    # Convert it to DF
    ##count_matrix_df = pd.DataFrame(count_matrix, index=df_indices)
    ##print(count_matrix_df)

    # Get the tf-idf matrix
    tfdif = TfidfVectorizer()
    tfidf_matrix = tfdif.fit_transform(df['categories']).todense()
    #print(tfidf_matrix)
    # Convert it to DF
    tfid_matrix_df = pd.DataFrame(tfidf_matrix, index=df_indices)
    #print(tfid_matrix_df)

    #print(tfid_matrix_df.loc[1].sort_values(ascending=False))

    #print()

    # Get rhe cosine similarity matrix
    ##cosine_sim_matrix = cosine_similarity(count_matrix, count_matrix)
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # Convert if as DF
    cosine_sim_matrix_df = pd.DataFrame(cosine_sim_matrix,
                                        index=df_indices,
                                        columns=df_indices)
    #print(cosine_sim_matrix_df)

    #print(cosine_sim_matrix_df.loc[1].sort_values(ascending=False)[1:10])
    #print(df.loc[1]['categories'])
    #print(df.loc[706]['categories'])
    #print(df.loc[2028]['categories'])

    data = [tfid_matrix_df, cosine_sim_matrix_df]

    save_object(data, out_filename)