# U.S. Senators: A Voting Pattern Study
**NOTE:** Please create a file named "api_key.txt" in the project root folder containing your Prorepublica Congress API Key as a line before running the notebook
## Introduction 
Predicting how Congressional legislators vote is important for understanding their past and future behavior. In this project, we explore U.S. senators voting patterns using a network-based approach. 

### Research questions
1. Can we predict the political stances of individuals as well as which senators are the closest to their ideology ?
2. Can a specific subset of senators, called the swing votes, be used to accurately predict the outcome of a vote ? 

## Notebooks
* **Main:** contains a step by step demonstration of the main results shown in the report.
* **Feature-Engineering:**  contains all the code used to process the raw data into a format that is convenient to work with
* **Data-Acquisition:** contains all the code used to retrieve the dataset from ProPublica's congres API

## Helper functions
Our project required some boilerplate code, especially for graph visualization that we chose to factorize in helper modules. Their contents are briefly summarized below.

* **cluster_utils.py** Helper functions for running the KMean algorithm.
* **plot_utils.py** All the boilerplate code required for creating the graph and the embedding vizualizations.
* **request_utils.py** Code fetching bill and roll call vote information from the relevant U.S governmental websites.
* **transductive_learning_utils.py** Code for running the variational minimization problem.
* **utils.py** Remaining utility functions that didn't fall into one of the previous categories
