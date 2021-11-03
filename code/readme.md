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

Note:
 - We currently have lines in ``biQQLSTM.py`` for you to tune the hypers, feel free to play with them.
 - We adapted some code from neuronblocks and quaternion networks (credits to them and big thank you). To check their source code, please visit their original github page and install necessary packages.
 - The txt file is a list of protected attributes from prior works by other researchers, please refer to the paper where there are our full citations.
