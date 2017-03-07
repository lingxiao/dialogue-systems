Introduction.

In this project we trained a vanilla seq-to-seq model on the twitter chat log found here:
https://github.com/suriyadeepan/datasets/tree/master/seq2seq/twitter. 

After tokenization (details below), there were 220754 question-answer pairs, 90% of which was used for training set,
the remaining 10% is used for test set.

Tokenization.

We used two tokenization schemes, nltk's vanilla tokenization scheme, which includes:
	- lowercase all tokens
	- removing all non alphanumeric characters 

and tworkenize found in tworkenize.py, this include:
	-


Model.


Experiments.