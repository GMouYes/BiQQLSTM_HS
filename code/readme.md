Assumptions:
 - You have one gpu as cuda:0.
 - You have the required packages installed. (check all the imports before you start running the code plz, expecially the NeuronBlocks and Pytorch-Quaternion-Neural-Networks, then modify the path in ``qqrnn.py``)
 - You splitted the data in data.csv and modified the path in data.py, redirected them to your desired files.
 - You would like the random seed as 1. 
 - You would like the outputs redirected to results.out

Run the code with:
 - ``CUDA_VISIBLE_DEVICES=0 nohup python3 -u biQQLSTM.py 1 > results.out &``

Final outcome:
 - Results will be printed in ``results.out``
 - Model will be stored in ``model.pkl``
 - Hypers will be stored in ``hyper.pkl``
 - ground truth and predictions are in two separate npy files.

Our best hypers are shown below:<br>
![Hyperparameters](https://github.com/GMouYes/BiQQLSTM_HS/blob/main/code/hypers.jpg)

Note:
 - We currently have lines in ``biQQLSTM.py`` for you to tune the hypers, feel free to play with them.
 - We adapted some code from neuronblocks and quaternion networks (credits to them and big thank you). To check their source code, please visit their original github page and install necessary packages.
 - The txt file is a list of protected attributes from prior works by other researchers, please refer to the paper where there are our full citations.

This model is tested against the following baseline models (please check the paper for full reference):<br>
```
Davidson'17: 
- This model used the linear support vector classifier trained by TPO features (i.e., TF-IDF and POS) and other text features such as sentiment scores, readability scores, and count indicators for \# of mentions and \# of hashtags, etc.

Kim'14: 
- We implemented Kim's Text-CNN for hate speech detection. This is a general-purpose classification framework widely applied in many text classification tasks.

Badjatiya'17:
- It is a domain specific LSTM-based network for detecting hate speeches.

Waseem'16:
- A logistic regression model trained on the bag of words features.

Zhang'18:
- The authors proposed the C-GRU model, which combines CNN and GRU to not only fit in small-scale data but also accelerate the whole progress. Such a framework is domain-specific to hate speech detection.

Indurthi'19:
- Fermi was proposed for Task 5 of SemEval-2019: HatEval: Multilingual Detection of Hate Speech Against Immigrants and Women on Twitter. \textbf{It was ranked the first in Subtask A for English}. They used pre-trained Universal Sentence Encoder followed by SVM.

BERT + LR:
- We used BERT base as word embedding, pooled the results, and then fed them to a logistic regression model for classification to reveal the embedding's contribution.

BERT CLS:
- Another variation in utilizing BERT directly is to finetune BERT through its [CLS] tag. Intuitively, finetuning the whole language model may gain better performance but is more expensive than freezing the embedding and only finetune the last layer (BERT + LR).
```
