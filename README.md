# Cascading Category Recommender (CCRec)
This project aims to perform category-level recommendation using a cascading approach
## Input Format
The input dataset should be a pandas dataframe with three columns -- past_topic, future_topic, and past_leaf.
- past_topic - a list object of the indices of past categories ranging from 0 to number of categories
- future_topic - a list object of the indices of future categories ranging from 0 to number of categories
- past_leaf - a list object of multiple lists. i<sup>th</sup> list contains all the item ids from the i<sup>th</sup> category in the past. 

Two examples of datasets, namely RocketRetail and Tmall, are given in the dataset folder.
## Requirements
- Linux
- NVIDIA GPU
- Pytorch 1.12+
- CUDA 11.8+
- Pandas 2.2+
## Training
The notebook and .py files give an example of training and testing on the Tmall dataset. By changing the dataset name and output file names, the result of RocketRetail can also be reproduced.
First, run `mle.py` to train the MLE model, which is the first module in the paper
```bash
python3 mle.py
```
Three files will be generated
- MLE.pkl: The MLE model parameters after training
- training_negative.pkl: The negative samples for training
- testing_negative.pkl: The candidate list for testing

Next, go to the Notebook folder and run the code in `Item-levelDependentVAE.ipynb` to get the embedding from VAE. One file will be generated, which is `topic_items_to_emb-tmall.pkl`

Finally, run `finalmodel.py` to train our final prediction model.
```bash
python3 finalmodel.py
```

The final model, `final.pkl` , will be generated
## Testing
Run `test.py` to test the model. 
```bash
python3 test.py
```
