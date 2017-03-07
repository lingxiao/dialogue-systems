# INTRODUCTION

In this project we trained a vanilla seq-to-seq model on the twitter chat log found here:
https://github.com/suriyadeepan/datasets/tree/master/seq2seq/twitter. 

After tokenization (details below), there were 220754 question-answer pairs, 90% of which was used for training set,
the remaining 10% is used for test set.

# PREPROCESSING

We used two tokenization schemes. The first is nltk's vanilla tokenization scheme, which includes:

* lowercase all tokens
* removing all non alphanumeric characters.

Note in the last case if a nonalphanumeric character appears inside of a word, then it is removed from the word. For example. Punctuations are not removed.


Next we usedd tworkenize found in tworkenize.py, this include:

* stripping white space
* removing emojis that do not appear consecutively with no space in between
* lower case
* split off edge punctuation

See tworkenize.py for a comprehensive list of tokenization steps. 

Finally, we removed any tweet questions-response pairs where the question is longer than 20 tokens, and the question is shorter than 3 tokens or longer than 20 tokens. 

The vocabulary is limited to 6000 characters, all out of vocabulary (OOV) words are mapped to the token 'unk'.

# MODELS
We used vanilla sequence to sequence model with attention mechanism, first proposed by Cho et al. (https://arxiv.org/pdf/1406.1078.pdf). This model was originally designed for machine translation and is trained to maximize the probabilty of target sequence given input sequence, where the cost is cross entropy. The model maps in the input sequence into a hidden vector, where the attention mechanism controls how much hidden information will propogate forward.

We utilized the orginial "translate.py" in the tensorflow repository written for French-English translation. All functions in the original file have been slightly modified for our twitter chatbot. In total, we trained three different models as follow:
	- preprocessed with nltk's vanilla tokenization scheme
	- preprocessed with tworkenize.py
	- preprocessed with tworkenize.py & early stopping
The total vocabulary size for both questions and answers is 6004 including the default special vocabulary used in Seq2Seq model (_PAD_, _UNK_, _GO_, _EOS_). We used four buckets with the following bucket sizes (question, answer)-pairs: [(5, 5), (10, 10), (15, 15), (25, 25)].

# TEST RESULTS and DISCUSSION 
We tested each model with the same set of sentences including some sentences in the training data. The results are in the "results" folder. Detailed explanantions are denoted below.

## 1) Preprocessed with NLTK's Vanilla Tokenization scheme
- learning rate: 0.1
- The total number of training steps: 107600
- Perplexity at the stopping point: 
	- Global: 27.09
	- Buckets: 122.35, 72.53, 70.51, 87.10
- Reuslt files:
	"nltk_107600.txt"
	"nltk_107600.png" (Just screenshot image)
: The global perplexity and the buckets' perplexities except the first one decreased as the steps increased, but the first bucket's perplexity decreased at the beginning and increased later. In the results, "unk" appears many times, even for the sentences in the training data.


## 2) Preprocessed with tworkenize.py
- learning rate: 0.5
- The total number of training steps: 128200
- Perplexity at the stopping point: 
	- Global: 1.11
	- Buckets: 2627.52, 16880.32, 43720.34, 24863.28
- Reuslt files:
	"tworken_128200.txt"
	"tworken_128200.png" (Just screenshot image)
: The global perplexity decreased up to 1.11 by the time we stopped training. However, each bucket's perplextiy kept increasing and became very large. Therefore, we trained another model stopping early - the third model. In terms of its result, it works better than the vanila tokenization case. Most of the answers do not completely make sense, but it generates less "unk" and more relevant words.

## 3) Preprocessed with tworkenize.py & early stopping
- learning rate: 0.5
- The total number of training steps: 27000
- Perplexity at the stopping point:
	- Global: 25.12
	- Buckets: 49.00, 48.75, 79.21, 92.70
- Reuslt files:
	"tworken_earlystopping_27000.txt"
	"tworken_earlystopping_27000.png" (Just screenshot image)
: The global perplexity at the stopping point of this model is higher than the one of the previous model. All buckets' perplexities of this model are much lower than the ones of the previous model. In the test results, the model generated "unk" more times than the first model. We didn't evaluated the results with a certain metrics, but this model's result apparently showed the worst performance among the three models.



## FUTURE WORK ##

Comparing those three different models, we found that good torkenization can improve performance of models. Also, the global perplexity matters more than each bucket's perplexity. Most of the answers in the test of the second model are grammatically not too wrong although we didn't give any grammar information when training. However, we think giving grammar information or using pre-trained, model which is grammatically correct, can improve the performance.



