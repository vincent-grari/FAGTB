# Fair Adversarial Gradient Tree Boosting

This work has been accepted as an oral presentation at IEEE ICDM 2019.

https://arxiv.org/abs/1911.05369

Fair classification has become an important topic in machine learning research.
While most bias mitigation strategies focus on neural networks, we noticed a lack of work
on fair classifiers based on decision trees even though they have proven very efficient.
In an up-to-date comparison of state-of-the-art classification algorithms in tabular data,
tree boosting outperforms deep learning. For this reason, we have developed a novel approach
of adversarial gradient tree boosting. The objective of the algorithm is to predict the output
Y with gradient tree boosting while minimizing the ability of an adversarial neural network 
to predict the sensitive attribute S. The approach incorporates at each iteration the gradient
of the neural network directly in the gradient tree boosting. We empirically assess our approach 
on 4 popular data sets and compare against state-of-the-art algorithms. The results show that 
our algorithm achieves a higher accuracy while obtaining the same level of fairness, as measured
using a set of different common fairness definitions.
![alt text](https://github.com/vincent-grari/FAGTB/blob/master/FAGTB.png?raw=true)

## Requirements
 1. Install the fairness library: pip3 install fairness
 2. Modify the file "ProcessedData.py" in the fairness library (/usr/local/lib/python3.6/dist-packages/fairness/data/objects/ProcessedData.py) with the file in this github repository.
    It requires for reproductibility results to have 80% for training and 20% for test set.
    
