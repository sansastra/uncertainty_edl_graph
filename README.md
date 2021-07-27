# uncertainty_edl_graph
Leveraging Evidential Deep Learning Uncertainties with Graph-based Clustering to Detect Anomalies (paper under submission)

Link: https://arxiv.org/pdf/2107.01557.pdf

Repository information:
1. Classification: contains source codes for the training and testing of an EDL classifier[1]. It is used to detect anomalous or normal samples for the loss of AIS signal detection and unusual turn detection 
2. Clustering: contains source codes for a graph-based clustering approach, utilizing DBSCAN and RDP algorithms, to build a graph based on AIS dataset, and associate a new AIS data to one of the edges of the graph.
3. Regression: contains source codes for training a recurrent  neural network together with Evidential layer[2] to forcast navigation data of vessels (longitude, latitude, speed-over-ground, course-over-ground).
4. Other codes are for supporting function and for preparing datasets.

[1] M. Sensoy et al, "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018. 
Link: https://papers.nips.cc/paper/2018/file/a981f2b708044d6fb4a71a1463242520-Paper.pdf

[2] A. Amini et al, "Deep Evidential Regression," NeurIPS 2020. 
Link: https://papers.nips.cc/paper/2020/file/aab085461de182608ee9f607f3f7d18f-Paper.pdf

Python code: EDL classifier in Pytorch and RNN-EDL regressor model in Tensorflow, for version and dependencies, see the requirements in the repository.  
