'''
This script should be run in the task2 folder
This script creates a pd.DataFrame with paper_ids and text. The text is cleaned from its references and citations.
'''

import pandas as pd
import os
import json
import numpy as np
# Removes all cite spans and ref spans including the brackets
# INPUT: json_file['body_text'][i]
# OUTPUT: plain text in a string without any references or annotations
def removeCitsRefs(json_paragraph):
    ranges = []
    places = []

    places.extend(json_paragraph['cite_spans'])
    places.extend(json_paragraph['ref_spans'])

    # Get ranges to delete
    for d in places:
        tmp = (d['start'], d['end'])
        ranges.append(tmp)
    text_list = list(json_paragraph['text'])
    for plc in ranges:
        for i in range(plc[0], plc[1]):
            text_list[i] = " "
    text = ''.join(text_list)
    return text


# INPUT: json_file['body_text']
# OUTPUT: Plain Text string
def getText(json_body):
    main_txt = ""
    for i in range(len(json_body)):
        tmp_txt = removeCitsRefs(json_body[i])
        tmp_txt += "\n"
        main_txt += tmp_txt
    return main_txt


# PaperId, Title + Body,
def getFileText(json_file):
    body = json_file['metadata']['title'] + " \n\n" + getText(json_file['body_text'])
    paper_id = json_file['paper_id']

    return [paper_id, body]


def createDataFrame(json_paths):
    files = []
    for j in range(len(json_paths)):
        # load json
        with open(json_paths[j], 'r') as f:
            article_json = json.load(f)
        files.append(getFileText(article_json))
    df = pd.DataFrame(files, columns=['paper_id', 'text'])
    return df


# Get data
# metadata = pd.read_csv("./data/metadata.csv", low_memory=False)
path_csv = "./data/papers19-20.csv"
path_metadata = './data/metadata.csv'

#Get metadata
metadata = pd.read_csv(path_metadata,low_memory=False)
#Drop all rows that doesn't have sha identifier and publish time
to_drop = list(metadata[pd.isna(metadata['publish_time'])].index)
to_drop = to_drop + list(metadata[pd.isna(metadata['sha'])].index)
to_drop = np.array(to_drop)
to_drop = list(np.unique(to_drop))
metadata = metadata.drop(to_drop,axis=0)
print("Got metadata")

#Filter only rows with publish_time either 2019 or 2020
p19 = metadata[metadata['publish_time'].str.contains('2019',regex=False)]
p20 = metadata[metadata['publish_time'].str.contains('2020',regex=False)]
metadata_papers = pd.concat([p19,p20],axis=0)
print("Got metadata from 2019 and 2020")

files = pd.DataFrame(columns=["name", "path"])

#Get JSON Files' paths
for dirname, _, filenames in os.walk('./data'):
    for filename in filenames:
        files = files.append({'name': filename, 'path': os.path.join(dirname, filename)}, ignore_index=True)
print("Got file paths")

#Drop irrelevant files that are not json
idx = files[files.name.str.contains(".json") == False].index
files.drop(index=idx, inplace=True)
files.reset_index(inplace=True)
files.drop("index", axis=1, inplace=True)
print("Got all file paths")
#Erase extensions
def getName(string):
    for i in range(len(string)):
        if string[i] == ".":
            return string[:i]
files['file_name'] = files['name'].map(lambda file_name: getName(file_name))
print("Delete extensions")

# Mark papers from 2019 or 2020
files['sha_flag'] = files['file_name'].isin(list(metadata_papers['sha']))
path_files = list(files[files['sha_flag']]['path'])


print("Cleaned paths without .json extension")
df_text = createDataFrame(path_files)
print("Created Dataframe")

df_text.to_csv(path_csv,index=False)
print("Saved DataFrame to ",os.path.abspath(path_csv))
