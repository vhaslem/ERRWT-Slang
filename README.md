Using the Edge Reinfored Random Walk with Triggering (ERRWT) Model developed in Di Bona et al. 2025, this code uses kaggle twitter data to produce a Rank-Frequency Diagram of the words (Zipf's Law), but partitions the words into slang and nonslang using the model.

Multiple restrictions have been made to ensure the code runs more efficiently. This includes restricting the graph size and only producing the top 5000 selections for each csv. 
Note: Continued work will also test Heaps' Law and potentially the brevity law.

Work needs to be done to introduce a temporal element

Kaggle Data Source and Citation: https://www.kaggle.com/datasets/kazanova/sentiment140
