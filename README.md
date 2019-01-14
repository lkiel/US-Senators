# Political affiliation identification and vote prediction on U.S. senators

## Introduction 
Predicting how Congressional legislators vote is important for understanding their past and future behavior. In this project, we explore U.S. senators voting patterns using a network-based approach. Our main goal is to use Congress roll call votes results on bills to predict political stances of individuals and voting outcome based on a restricted set of features. The latter will consist of carefully chosen votes which are designed to convey the highest information on senators voting behaviour. We will also show that we can accurately predict the outcome of a vote only by looking at a small, well-chosen subset of senators’ voting positions. 

## Notebooks
* **Main:** contains a step by step demonstration of the main results shown in the report.
* **Feature-Engineering:**  contains all the code used to process the raw data into a format that is convenient to work with
* **Data-Acquisition:** contains all the code used to retrieve the dataset from ProPublica's congres API

## Helper functions
Our project required some boilerplate code, especially for graph visualization that we chose to factorize in helper modules. Their contents are briefly summarized below.

* **cluster_utils.py** 
* **plot_utils.py** All the boilerplate code required for creating the graph and the embedding vizualizations.
* **request_utils.py** Code fetching bill and roll call vote information from the relevant U.S governmental websites.
* **transductive_learning_utils.py** Code for running the variational minimization problem.
* **utils.py** Remaining utility functions that didn't fall into one of the previous categories
