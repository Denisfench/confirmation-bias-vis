#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

import nltk
import tweepy
from dotenv import load_dotenv
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from datetime import datetime
import re
import cv2
import numpy as np
import pandas as pd
import pickle
# NOTE: THE WORDCLOUD PACKAGE ISN'T WORKING FOR SOME VERSIONS OF PYTHON.
# LOCKING PYTHON VERSION TO 3.7 BECAUSE OF THAT.
from wordcloud import WordCloud
import spacy
import spacy_transformers
import torch


# In[2]:


import ssl
def set_up_ssl():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

set_up_ssl()


# In[3]:


load_dotenv()
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET_KEY = os.getenv('TWITTER_API_SECRET_KEY')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')


# In[4]:


print("Authenticating to Twitter...")

client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)
auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET_KEY)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)


# In[5]:


twitter_russia_sources_rus_usernames = ["@1tvru_news", "@ru_rbc",
                                         "@er_novosti",
                              "@rt_com",
                              "@medvedevrussia", "@kremlinrussia",
                              "@rentvchannel", "@vesti_news", "@kpru"]

twitter_ukraine_sources_rus_usernames = ["@dmitry_gordon", "@SvobodaRadio",
                               "@euronewsru", "@FeyginMark4", "@tvrain", "@teamnavalny"]

twitter_ukraine_sources_ukr_usernames = ["@HromadskeUA", "@tsnua", "@24tvua", "@unian",
                               "@radiosvoboda", "@5channel", "@EspresoTV"]

twitter_ukraine_sources_eng_usernames = ["@mschwirtz", "@KyivIndependent", "@KyivPost",
                               "@mchancecnn", "@fpleitgenCNN", "@Kasparov63",
                               "@ikhurshudyan", "@myroslavapetsa",
                               "@langfittnpr", "@ElBeardsley", "@timkmak"]


# https://dev.to/twitterdev/a-comprehensive-guide-for-using-the-twitter-api-v2-using-tweepy-in-python-15d9

# In[6]:


# THIS CELL HITS TWITTER API

def get_user_id_from_username(username):
    user = api.get_user(screen_name=username)
    return user.id


# In[7]:


russia_sources_rus = []
ukraine_sources_rus = []
ukraine_sources_ukr = []
ukraine_sources_eng = []


# In[8]:


LOAD_CLUSTERS_DATA = True


# In[9]:


if not LOAD_CLUSTERS_DATA:
    for username in twitter_russia_sources_rus_usernames:
        russia_sources_rus.append((username, get_user_id_from_username(username)))

    for username in twitter_ukraine_sources_rus_usernames:
        ukraine_sources_rus.append((username, get_user_id_from_username(username)))

    for username in twitter_ukraine_sources_ukr_usernames:
        ukraine_sources_ukr.append((username, get_user_id_from_username(username)))

    for username in twitter_ukraine_sources_eng_usernames:
        ukraine_sources_eng.append((username, get_user_id_from_username(username)))


# In[10]:


if not LOAD_CLUSTERS_DATA:
    russia_sources_rus_pickled = pickle.dumps(russia_sources_rus)
    ukraine_sources_rus_pickled = pickle.dumps(ukraine_sources_rus)
    ukraine_sources_ukr_pickled = pickle.dumps(ukraine_sources_ukr)
    ukraine_sources_eng_pickled = pickle.dumps(ukraine_sources_eng)


# In[11]:


CLUSTERS_SERIALIZATION_DIR = "data_clusters/"


# In[12]:


if not LOAD_CLUSTERS_DATA:
    print("Writing pickled data to a file...")

    with open(CLUSTERS_SERIALIZATION_DIR + 'russia_sources_rus_pickled.pickle', 'wb') as f:
        f.write(russia_sources_rus_pickled)

    with open(CLUSTERS_SERIALIZATION_DIR + 'ukraine_sources_rus_pickled.pickle', 'wb') as f:
        f.write(ukraine_sources_rus_pickled)

    with open(CLUSTERS_SERIALIZATION_DIR + 'ukraine_sources_ukr_pickled.pickle', 'wb') as f:
        f.write(ukraine_sources_ukr_pickled)

    with open(CLUSTERS_SERIALIZATION_DIR + 'ukraine_sources_eng_pickled.pickle','wb') as f:
        f.write(ukraine_sources_eng_pickled)


# In[13]:


if LOAD_CLUSTERS_DATA:
    with open(CLUSTERS_SERIALIZATION_DIR + 'russia_sources_rus_pickled.pickle',\
            'rb') as f:
        russia_sources_rus = pickle.load(f)

    with open(CLUSTERS_SERIALIZATION_DIR + 'ukraine_sources_rus_pickled' \
            '.pickle', 'rb') as f:
        ukraine_sources_rus = pickle.load(f)

    with open(CLUSTERS_SERIALIZATION_DIR + 'ukraine_sources_ukr_pickled' \
            '.pickle', 'rb') as f:
        ukraine_sources_ukr = pickle.load(f)

    with open(CLUSTERS_SERIALIZATION_DIR + 'ukraine_sources_eng_pickled' \
            '.pickle', 'rb') as f:
        ukraine_sources_eng = pickle.load(f)


# In[14]:


# THIS CELL HITS TWITTER API

def get_user_followers(user_name, user_id, num_pages=1, num_followers=10):
    followers = []
    # User rate limit (User context): 15 requests per 15-minute window per each authenticated user
    # limit – Maximum number of requests to make to the API
    # max_results : The maximum number of results to be returned per page. This can be a number between 1 and the 1000. By default, each page will return 100 results.
    # I.E. 15 000 followers can be returned in 15 minutes
    followers_paginator = tweepy.Paginator(client.get_users_followers, id =
    user_id, max_results = num_followers, limit = num_pages).flatten()
    for follower in followers_paginator:
        followers.append(follower)
    return (user_name, user_id), followers


# In[15]:


# THIS CELL HITS TWITTER API

def get_user_follower_count(user_id):
    # fetching the user
    user = api.get_user(user_id = user_id)
    return user.followers_count


# In[16]:


rus_cluster_followers = []
ukr_eng_cluster_followers = []
ukr_rus_cluster_followers = []
ukr_ukr_cluster_followers = []


# In[17]:


# PROBLEM: pickling followers causes recursion depth exceeded problem
# SOLUTION: process the followers data right away and write the result in csv
# format

LOAD_FOLLOWERS = False
FOLLOWERS_DIR = "followers/"


# In[18]:


# PRODUCTION CODE
# get 105 000 followers per cluster: 1:45 min per cluster, 105 requests

'''
if  not LOAD_FOLLOWERS:
    for cluster in russia_sources_rus:
        rus_cluster_followers.append(get_user_followers(cluster[0], cluster[1]))

    for cluster in ukraine_sources_rus:
        ukr_eng_cluster_followers.append(get_user_followers(cluster[0], cluster[1]))

    for cluster in ukraine_sources_ukr:
        ukr_rus_cluster_followers.append(get_user_followers(cluster[0], cluster[1]))

    for cluster in ukraine_sources_eng:
        ukr_ukr_cluster_followers.append(get_user_followers(cluster[0], cluster[1]))
'''


# In[19]:


# THIS CODE HITS TWITTER API
# TEST CODE: 50 followers per cluster, 2 cluster per each
if not LOAD_FOLLOWERS:
    for i in range(0, 2):
        rus_cluster_followers.append(get_user_followers(russia_sources_rus[i][0],                                               russia_sources_rus[i][1]))

    for i in range(0, 2):
        ukr_eng_cluster_followers.append(get_user_followers(ukraine_sources_rus[i][0],                                               ukraine_sources_rus[i][1]))


# In[20]:


type(rus_cluster_followers)


# In[21]:


if not LOAD_FOLLOWERS:
    rus_cluster_followers_pickled = pickle.dumps(rus_cluster_followers)
    ukr_eng_cluster_followers_pickled = pickle.dumps(ukr_eng_cluster_followers)
    ukr_rus_cluster_pickled = pickle.dumps(ukr_rus_cluster_followers)
    ukr_ukr_cluster_pickled = pickle.dumps(ukr_ukr_cluster_followers)


# In[22]:


if not LOAD_FOLLOWERS:
    print("Writing pickled data to a file...")

    with open(FOLLOWERS_DIR + 'rus_cluster_followers.pickle', 'wb') as f:
        f.write(rus_cluster_followers_pickled)

    with open(FOLLOWERS_DIR + 'ukr_eng_cluster_followers.pickle', 'wb') as f:
        f.write(ukr_eng_cluster_followers_pickled)

    with open(FOLLOWERS_DIR + 'ukr_rus_cluster_followers.pickle', 'wb') as f:
        f.write(ukr_rus_cluster_pickled)

    with open(FOLLOWERS_DIR + 'ukr_ukr_cluster_followers.pickle','wb') as f:
        f.write(ukr_ukr_cluster_pickled)


# In[23]:


# NOTE: causes recursion depth exceeded problem

if LOAD_FOLLOWERS:

    with open(FOLLOWERS_DIR + 'rus_cluster_followers.pickle',\
            'rb') as f:
        rus_cluster_followers = pickle.load(f)

    with open(FOLLOWERS_DIR + 'ukr_eng_cluster_followers' \
            '.pickle', 'rb') as f:
        ukr_eng_cluster_followers = pickle.load(f)

    with open(FOLLOWERS_DIR + 'ukr_rus_cluster_followers' \
            '.pickle', 'rb') as f:
        ukr_rus_cluster_followers = pickle.load(f)

    with open(FOLLOWERS_DIR + 'ukr_ukr_cluster_followers' \
            '.pickle', 'rb') as f:
        ukr_ukr_cluster_followers = pickle.load(f)


# In[24]:


print(ukr_eng_cluster_followers)


# In[25]:


'''
print(type(followers[('@minregion_ua', 3333475643)][0]))
print(followers[('@minregion_ua', 3333475643)][0].name)
print(followers[('@minregion_ua', 3333475643)][0].id)
'''
type(rus_cluster_followers[0])


# In[26]:


CLUSTER_IDX = 0
FOLLOWER_IDX = 1

rus_cluster = {}
ukr_eng_cluster = {}
ukr_rus_cluster = {}
ukr_ukr_cluster = {}


# In[27]:


def cluster_to_dict(cluster_list):

    cluster_dfs = {}

    for cluster in cluster_list:
        follower_names = [follower.name for follower in cluster[FOLLOWER_IDX]]
        follower_ids = [follower.id for follower in cluster[FOLLOWER_IDX]]
        followers_data = {
            'username': follower_names,
            'user_id': follower_ids
        }
        cluster_dfs[cluster[CLUSTER_IDX]] = pd.DataFrame(followers_data)

    return cluster_dfs


# In[28]:


rus_cluster = cluster_to_dict(rus_cluster_followers)
ukr_eng_cluster = cluster_to_dict(ukr_eng_cluster_followers)
ukr_rus_cluster = cluster_to_dict(ukr_rus_cluster_followers)
ukr_ukr_cluster = cluster_to_dict(ukr_ukr_cluster_followers)


# In[29]:


# this fucntion converts a list of cluster centers and their followers to csv
# files
CLUSTERS_DIR = "clusters/"

def clusters_to_files(clusters_df):
    for cluster_center, cluster_followers in clusters_df.items():
        cluster_csv = cluster_followers.to_csv()
        with open(CLUSTERS_DIR + cluster_center[CLUSTER_IDX][1:] + '_' + str(cluster_center[FOLLOWER_IDX]) + '.csv',
                  'w') as f:
            f.write(cluster_csv)


# In[30]:


LOAD_CLUSTERS = True


# In[31]:


if not LOAD_CLUSTERS:
    clusters_to_files(rus_cluster)
    clusters_to_files(ukr_eng_cluster)


# In[32]:


russia_sources_rus


# In[33]:


if LOAD_CLUSTERS:
    print("Loading data for rus_cluster")
    rus_cluster = {}
    for cluster in russia_sources_rus:
        path_to_file = CLUSTERS_DIR + cluster[0][1:] + '_' + str(cluster[1]) + \
 '.csv'
        if os.path.exists(path_to_file):
            cluster_followers = pd.read_csv(path_to_file,
                                            usecols=['username', 'user_id'], dtype={'user_id': int})
            rus_cluster[cluster] = cluster_followers

    print("Loading data for ukr_eng_cluster")
    ukr_eng_cluster = {}
    for cluster in ukraine_sources_eng:
        path_to_file = CLUSTERS_DIR + cluster[0][1:] + '_' + str(cluster[1]) + \
 '.csv'
        if os.path.exists(path_to_file):
            cluster_followers = pd.read_csv(path_to_file,
                                            usecols=['username', 'user_id'], dtype={'user_id': int})
            ukr_eng_cluster[cluster] = cluster_followers

    print("Loading data for the ukr_rus_cluster")
    ukr_rus_cluster = {}
    for cluster in ukraine_sources_rus:
        path_to_file = CLUSTERS_DIR + cluster[0][1:] + '_' + str(cluster[1]) + \
 '.csv'
        if os.path.exists(path_to_file):
            cluster_followers = pd.read_csv(path_to_file,
                                            usecols=['username', 'user_id'],
                                            dtype={'user_id': int})
            ukr_rus_cluster[cluster] = cluster_followers

    print("Loading data for the ukr_ukr_cluster")
    ukr_ukr_cluster = {}
    for cluster in ukraine_sources_ukr:
        path_to_file = CLUSTERS_DIR + cluster[0][1:] + '_' + str(cluster[1]) + \
 '.csv'
        if os.path.exists(path_to_file):
            cluster_followers = pd.read_csv(path_to_file,
                                            usecols=['username', 'user_id'], dtype={'user_id': int})
            ukr_ukr_cluster[cluster] = cluster_followers


# In[34]:


rus_cluster


# <h2>Visualizing Connections as a Graph</h2>

# In[35]:


ukr_rus_cluster


# In[36]:


import networkx as nx


# In[37]:


print("Creating a DataFrame containing the complete graph")

GROUP_ID_MAP = {
    "rus_cluster": 0,
    "ukr_eng_cluster": 1,
    "ukr_rus_cluster": 2,
    "ukr_ukr_cluster": 3
}

global_graph_pd = pd.DataFrame(columns=['username', 'user_id',
                                     'cluster_name', 'cluster_id',
                                     'cluster_follow_count', 'group_id'])


# In[38]:


def add_cluster_to_global_graph(cluster, group_name, graph_pd):
    curr_cluster_df = pd.DataFrame()
    for cluster_center, cluster_followers in cluster.items():
        curr_cluster_df = cluster_followers.copy(deep=True)
        curr_cluster_df.insert(2, "cluster_name", cluster_center[0])
        curr_cluster_df.insert(3, "cluster_id", int(cluster_center[1]))
        curr_cluster_df.insert(4, "cluster_follow_count",
                               get_user_follower_count(cluster_center[1]),
                               True)
        curr_cluster_df.insert(5, "group_id", GROUP_ID_MAP[group_name])
    return pd.concat([graph_pd, curr_cluster_df])


# In[39]:


def save_image(img, img_name):
    IMG_DIR = "visualizations/"
    plt.imsave(IMG_DIR + img_name, img)


# In[40]:


global_graph_pd = global_graph_pd.iloc[0:0]
global_graph_pd = add_cluster_to_global_graph(rus_cluster, "rus_cluster", global_graph_pd)
global_graph_pd = add_cluster_to_global_graph(ukr_rus_cluster, "ukr_rus_cluster",
                                              global_graph_pd)


# In[41]:


print(global_graph_pd.size)
print(set(global_graph_pd['cluster_name']))
global_graph_pd.head()
print(global_graph_pd.size)


# In[55]:


global_graph_pd = global_graph_pd.astype({
    "username" : str,
    "user_id" : int,
    "cluster_name" : str,
    "cluster_id" : int,
    "cluster_follow_count" : int,
    "group_id" : int
})
global_graph_pd_columns = list(global_graph_pd.columns)
print(global_graph_pd.dtypes)
print(global_graph_pd_columns)
print(global_graph_pd.head())


# In[56]:


print("Constructing NetworkX graph")

# what if you store all attributes as edge attributes?
G = nx.from_pandas_edgelist(global_graph_pd, source='username',
                            target='cluster_name', edge_attr = global_graph_pd_columns,
                             create_using=nx.DiGraph())

pos = nx.spring_layout(G)


# In[58]:


print("Visualizing a small subset of connections...")

subgraph = G.subgraph(list(G.nodes)[:10])

subgraph_pos = nx.spring_layout(subgraph)

nx.draw_networkx(subgraph, subgraph_pos)


# <h2>Network Anslysis Questions:</h2>
# 
# 1. How many connections are overlapping within each given group. (e.g. how
# many people following SWJ also follow NYT)
# 2. How many overlaps are there in between clusters from different groups?
# 
# 3. Do nodes cluster into tightly connected groups?
# 
# <h2>Network Visualization Questions</h2>
# 
# 3. Visualize the connections in pretty way
# 4. Visualize Cluster sizes
# 5. Visualize groups by color coding them
# 
# <h2>Sentiment Analysis Questions</h2>
# 1. Word Cloud: what are people within each group discussing (Use entity recognition)?
# 
# 2. Tag Cloud: who are people within each group discussing?
# 
# 3. What sentiment do mentions within each groups have?
# 
# 4. What is a general Twitter sentiment on this topic (Can use Ukraine dataset)
# 
# <h2>Visualization Techniques that can be leveraged</h2>
# 1. Network Visualization
# 
# 2. Coloring nodes
# 
# 3. Coloring Connections
# 
# 4. Size of the nodes
# 
# 5. Shape of the nodes
# 

# In[59]:


def get_edge_attributes(graph, attr_list):
    KEY = 0
    VALUE = 1
    edge_attrs = {}
    for attr in attr_list:
        edges_attribute = nx.get_edge_attributes(graph, attr)
        for edge_attr in edges_attribute.items():
            if edge_attr[KEY] in edge_attrs.keys():
                edge_attr_values = edge_attrs[edge_attr[KEY]]
                edge_attr_values.append(edge_attr[VALUE])
                edge_attrs[edge_attr[KEY]] = edge_attr_values
            else:
                edge_attrs[edge_attr[KEY]] = [edge_attr[VALUE]]
    return edge_attrs

edge_attrs = get_edge_attributes(subgraph, global_graph_pd_columns)

print(edge_attrs)


# In[60]:


print("Running analysis on the network...")

# https://subscription.packtpub.com/book/big-data-and-business-intelligence/9781789955316/7

G.edges


# In[61]:


subgraph.edges


# In[62]:


# probability some bad values are in dataframe that have incorrect type
# TODO: CREATE COMMUNITIES BY HAND, YOU KNOW WHAT THEY ARE

import networkx.algorithms.community as nxcom

# identifying communities within the network
twitter_communities = sorted(nxcom.greedy_modularity_communities(G,
                                                                 weight=None),
                             key=len,
                             reverse=True)


# In[63]:


print("The number of communities detected is ", len(twitter_communities))
print("The communities detected are ", twitter_communities)


# In[64]:


print("Visualizing Identified Communities")

def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1

def set_edge_community(G):
    '''Find internal edges and add their community to their attributes'''
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0


# In[65]:


def get_color(i, r_off=1, g_off=1, b_off=1):
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)


# In[66]:


# Set node and edge communities
set_node_community(G, twitter_communities)
set_edge_community(G)

# Set community color for nodes
node_color = [
    get_color(G.nodes[v]['community'])
    for v in G.nodes]

# Set community color for internal edges
external = [
    (v, w) for v, w in G.edges
    if G.edges[v, w]['community'] == 0]

internal = [
    (v, w) for v, w in G.edges
    if G.edges[v, w]['community'] > 0]

internal_color = [
    get_color(G.edges[e]['community'])
    for e in internal]


# In[67]:


# Draw external edges
nx.draw_networkx(
    G, pos=pos, node_size=0,
    edgelist=external, edge_color="#333333",
    alpha=0.2, with_labels=False)

# Draw internal edges
nx.draw_networkx(
    G, pos=pos, node_size=0,
    edgelist=internal, edge_color=internal_color,
    alpha=0.05, with_labels=False)


# In[68]:


all_clusters = []

all_clusters.extend(rus_cluster)
all_clusters.extend(ukr_eng_cluster)
all_clusters.extend(ukr_rus_cluster)
all_clusters.extend(ukr_ukr_cluster)

all_clusters_dict = {}

for cluster in all_clusters:
    if str(cluster[0]) in G.nodes:
        all_clusters_dict[str(cluster[0])] = str(cluster[0])


# In[69]:


# TODO: SAVE AS PDF

pos = nx.spring_layout(G, k=0.15)

G_node_degrees = dict(G.degree)

# Draw internal edges
nx.draw_networkx(
    G, pos=pos, node_size=[v * 1 for v in G_node_degrees.values()],
    edgelist=internal, edge_color=internal_color,
    alpha=0.05, with_labels=True, label='Group Follower Clusters',
    labels=all_clusters_dict, font_color='#00ff00', font_weight='bold')


# In[104]:


def sort_graph_by_deg_desc(graph):
    return sorted(G.degree, key=lambda x: x[1], reverse=True)


# In[105]:


G_deg_sorted = sort_graph_by_deg_desc(G)


# In[106]:


print("Performing connectivity analysis on a graph")

node_connections = nx.all_pairs_node_connectivity(subgraph)


# In[107]:


def get_shared_nodes_within_group(node_connections, group_members):
    shared_connections = []
    for username, connections_dict in node_connections.items():
        for group_member in group_members:
            if group_member in connections_dict.keys() and connections_dict[group_member] == 1:
                shared_connections.append(username)
    return shared_connections


# In[108]:


get_shared_nodes_within_group(node_connections, rus_cluster)


# In[109]:


def get_shared_nodes_between_groups(node_connections, cluster_groups):
    shared_connections = {}
    for username, connections_dict in node_connections.items():
        for group_name, group_members in cluster_groups.items():
            for group_member in group_members:
                if group_member in connections_dict.keys() and connections_dict[group_member] == 1:
                    if group_name not in shared_connections.keys():
                        shared_connections.append(username)
                    else:
                        shared_connections[group_name] = shared_connections[group_name].append(username)
    return shared_connections


# In[110]:


TWITTER_CLUSTER_GROUPS = {
    "rus_cluster": russia_sources_rus,
    "ukr_eng_cluster": ukraine_sources_rus,
    "ukr_rus_cluster": ukraine_sources_ukr,
    "ukr_ukr_cluster": ukraine_sources_eng
}


# In[111]:


shared_connections_between_groups = get_shared_nodes_between_groups\
(node_connections, TWITTER_CLUSTER_GROUPS)


# In[112]:


shared_connections_between_groups


# In[113]:


'''
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(10, 10))
plt.style.use('ggplot')
nodes = nx.draw_networkx_nodes(G, pos,
                               alpha=0.8)
nodes.set_edgecolor('k')
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.2)
'''


# In[114]:


# TODO: analyze social network graph using NetworkX
# TODO: perhaps the network should be cleaned of low degree connected users
#  for visualization purposes
# TODO: run the graph analytics on the original network w/o visualizing it


# <h2>Visualizing the Network using Gephi</h2>

# In[115]:


# TODO: node size should correspond to the number of followers the node has
# TODO: node color should correspond to the node group


# <h2>Visualizing the Network using Neo4j</h2>

# <h2>Analyzing group sentiments</h2>

# In[116]:


# THIS CELL HITS TWITTER API

# NOTE: THIS FUNCTION PULLS THE TWEETS MENTIONING A PARTICULAR USER, NOT FROM
# A PARTICULAR USER

def get_user_tweet_mentions(username, num_tweets, entities=None):

    search_query = username + " OR "

    if entities is not None:
        # include all possible candidate names in a query
        for entity in entities:
            search_query += "entity:" + '"' + entity + '"' + " OR "

    # remove the last OR statement
    search_query = search_query[:-3]

    search_query += "-is:retweet"

    tweets = api.search_tweets(q = search_query, count = num_tweets,
                               tweet_mode="extended")

    return tweets


# In[117]:


# THIS CELL HITS TWITTER API

def get_user_tweets(user_id, num_tweets):

    # https://docs.tweepy.org/en/stable/client.html#tweepy.Client.get_users_tweets
    # https://docs.tweepy.org/en/stable/expansions_and_fields.html#expansions-parameter
    # https://dev.to/twitterdev/a-comprehensive-guide-for-using-the-twitter-api-v2-using-tweepy-in-python-15d9

    tweets = client.get_users_tweets(id = user_id, max_results = num_tweets,
                               exclude = ['retweets'], expansions='entities.mentions.username')

    return tweets


# In[118]:


# Sentiment in relation to a different group

# Tweets word Cloud

# Tweets HashTags Word Cloud

# Tweets events mapping? (Maybe)


# In[119]:


# https://www.caida.org/catalog/software/walrus/


# In[120]:


print("Pulling a sample of tweets for all clusters")

NUM_TWEETS = 5
USERNAME_IDX = 0
USER_ID_IDX = 1

cluster_tweets = {}

for cluster in all_clusters:
    cluster_tweets[cluster] = get_user_tweets(cluster[USER_ID_IDX], NUM_TWEETS)


# In[121]:


cluster_tweets


# In[122]:


def convert_tweets_to_json(raw_tweets_dict):
    json_tweets = []
    for username, tweets in raw_tweets_dict.items():
        for tweet in tweets:
            json_tweet_str = json.dumps(tweet.text)
            json_tweet = json.loads(json_tweet_str)
            json_tweets.append(json_tweet)
    return json_tweets


# In[123]:


tweet_status = api.get_status(1598709448129662977)

print(tweet_status._json)


# In[124]:


cluster_tweets


# In[125]:


TWEETS_DIR = 'tweets/'

def tweets_to_df(raw_tweets_dict):
    tweets_df = pd.DataFrame(
        {
            'Cluster Username' : pd.Series(dtype='str'),
            'Cluster ID' : pd.Series(dtype='int'),
            'Tweet Text' : pd.Series(dtype='str')
        }
    )
    for username, tweets in raw_tweets_dict.items():
            for tweet in tweets.data:
                new_df_row = {
                    'Cluster Username' : username[0],
                    'Cluster ID' : username[1],
                    'Tweet Text' : str(tweet)
                }
                tweets_df = tweets_df.append(new_df_row, ignore_index=True)
    return tweets_df

def tweets_df_to_csv(tweets_df):
    tweets_df.to_csv(TWEETS_DIR + 'tweets.csv', index=False)
    return


# In[126]:


tweets_df = tweets_to_df(cluster_tweets)
tweets_df_to_csv(tweets_df)


# <h2>Processing Collected Tweets Before Visualizing them</h2>

# In[127]:


print("Use this function to clean the tweet's body")

def clean_tweet(tweet_body):
    # remove @ mentions from the tweet
    # text = re.sub(r'@[A-Za-z0-9]+', '', tweet_body)
    # remove the hashtags from tweets
    # text = re.sub(r'#', '', text)
    # remove retweet
    text = re.sub(r'RT[\s]+', '', tweet_body)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/\S+', '', text)
    return text


# In[128]:


import string

def remove_ua_stopwords(tweet_body):
    stopwords_ua = pd.read_csv("stopwords_ua.txt", header=None, names=['stopwords'])
    stop_words_ua = list(stopwords_ua.stopwords)
    text = "".join([word for word in tweet_body if word not in string.punctuation])
    text = text.lower()
    tokens = re.split('\W+', text)
    text = [word for word in tokens if word not in stop_words_ua]
    return text


# In[129]:


# this fucntion doesn't support Ukrainian

from nltk.corpus import stopwords

print("Supported langauges are: ")
print(stopwords.fileids())

def remove_stopwords(tweet_body, lang='english'):
    stop_words = set(stopwords.words(lang))
    word_tokens = nltk.word_tokenize(tweet_body)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    filtered_sentence =' '.join(filtered_sentence)
    return filtered_sentence


# In[130]:


txt = "Засвідчивши свою відпускну в петербурзькій Палаті цивільного суду, Шевченко став учнем Академії мистецтв, де його наставником став К. Брюллов.[49] За словами Шевченка: «настала найсвітліша доба його життя, незабутні, золоті дні» навчання в Академії мистецтв, яким він присвятив у 1856 році автобіографічну повість «Художник»."

txt = remove_ua_stopwords(txt)

print(txt)


# In[131]:


txt = "Окончательно фамилия «Достоевский» закрепилась за внуками Данилы Ивановича, потомки которых со временем становятся типичной служилой шляхтой[13][14]. Пинская ветвь Достоевских на протяжении почти двух веков упоминалась в различных документах, но со временем интегрировалась польско-литовским государством, утратив дворянство[15][16]. Во второй половине XVII века род перебирался на Украину. В это же время резко сократилось количество упоминаний фамилии в исторических документах[16]."

txt = remove_stopwords(txt, lang='russian')

print(txt)


# In[132]:


# to be used for word cloud creation
def get_cleaned_tokens(tweets_df, lang='english'):
    tweet_tokens_dict = {}
    for _, row in tweets_df.iterrows():
        tweet_text = row['Tweet Text']
        tweet_text = clean_tweet(tweet_text)
        # remove the stopwords
        if lang == 'ukrainian':
            tweet_text = remove_ua_stopwords(tweet_text)
        else:
            tweet_text = remove_stopwords(tweet_text, lang=lang)
        tokenized_tweet = nltk.word_tokenize(tweet_text)
        if row['Cluster Username'] in tweet_tokens_dict.keys():
            tweet_tokens_dict[row['Cluster Username']].extend(tokenized_tweet)
        else:
            tweet_tokens_dict[row['Cluster Username']] = tokenized_tweet

    return tweet_tokens_dict


# <h2>Visualizing Tweet Content using Word Cloud</h2>

# In[133]:


tweets_df_tokens = get_cleaned_tokens(tweets_df, lang='russian')

tweets_df_tokens


# In[134]:


# NOTE: wordcloud per cluster, not per group
def create_wordcloud(tweets_df_tokens, cluster_name):
    tweet_word_cloud = WordCloud(random_state=21,
                           max_font_size=119).generate(' '.join(tweets_df_tokens[cluster_name]))
    plt.figure(figsize=(20,10))
    plt.imshow(tweet_word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    return


# In[135]:


create_wordcloud(tweets_df_tokens, '@SvobodaRadio')


# <h2>Performing Entity Recognition and Sentiment Analysis for different
# groups of tweets</h2>

# In[136]:


print("Loading NLP libraries for English, Ukrainian and Russian languages")


# In[137]:


get_ipython().system('python3 -m spacy download en_core_web_sm')

nlp_eng = spacy.load('en_core_web_sm')


# In[138]:


get_ipython().system('python3 -m spacy download uk_core_news_sm')

nlp_ukr = spacy.load('uk_core_news_sm')


# In[139]:


get_ipython().system('python3 -m spacy download ru_core_news_sm')

nlp_rus = spacy.load('ru_core_news_sm')


# In[140]:


# Spacy has vocabulary for English, Ukrainian and Russian languages

text = ("Восемь вандалов, которые срезали со стены дома в #Гостомеле женщину "
        "в противогазе авторства #Бэнкси, дали показания полиции #Украина #россия #ВСУ #война #агрессияроссии #вторжениероссии #stoprussia # #войнасукраиной #Войнапутина")

doc = nlp_rus(text)

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text)
    print(entity.text, entity.label_)


# In[141]:


print("Performing sentiment analysis on tweets")

analyzer = SentimentIntensityAnalyzer()

analyzer.polarity_scores("Вейдер очень умный, красивый и смешной!!!")


# In[142]:


analyzer.polarity_scores("Новый ракетный удар по мирным жителям Донецка. В сети публикуют момент атаки. Украинские боевики открыли огонь по многоэтажке. Прямое попадание реактивного снаряда. Взрывной волной выбило стекла в соседних зданиях. Два человека погибли https://t.co/xUgOmV4iWN https://t.co/hN8uLeHwzu")


# <h2>Performing Entity and Sentiment Analysis</h2>

# In[158]:


def augment_entity_to_df(tweets_df, lang='english'):

    new_df = tweets_df.copy(deep=True)

    entities_col = []

    for _, row in tweets_df.iterrows():
        entities = []
        tweet_text = row['Tweet Text']
        tweet_text = clean_tweet(tweet_text)
        if lang == 'english':
            nlp_doc = nlp_eng(tweet_text)
        elif lang == 'russian':
            nlp_doc = nlp_rus(tweet_text)
        elif lang == 'ukrainian':
            nlp_doc = nlp_ukr(text)
        else: raise Exception('Language not supported.')

        # NOTE: enteteties can be filtered based on the entity. e.g. only
        # people
        for entity in nlp_doc.ents:
            entities.append(entity.text)

        entities = list(set(entities))

        entities_col.append(entities)

    new_df['Entities'] = entities_col

    return new_df


# In[159]:


new_df = augment_entity_to_df(tweets_df, lang='russian')
print(new_df)


# In[160]:


def augment_sent_to_df(tweets_df, lang='english'):

    new_df = tweets_df.copy(deep=True)

    analyzer = SentimentIntensityAnalyzer()

    neg_sent = []
    pos_sent = []

    for _, row in tweets_df.iterrows():
        tweet_text = row['Tweet Text']
        tweet_text = clean_tweet(tweet_text)
        tweet_sentiment_scores = analyzer.polarity_scores(tweet_text)
        neg_sent.append(tweet_sentiment_scores['neg'])
        pos_sent.append(tweet_sentiment_scores['pos'])

    new_df['Neg'] = neg_sent
    new_df['Pos'] = pos_sent

    return new_df


# In[161]:


new_df = augment_sent_to_df(new_df, lang='russian')
print(new_df)


# <h2>Creating Sentiment Visualizations in Plotly</h2>

# 

# In[ ]:




