Introduction.

In this project we trained a vanilla seq-to-seq model on the twitter chat log found here:
https://github.com/suriyadeepan/datasets/tree/master/seq2seq/twitter. 

After tokenization (details below), there were 220754 question-answer pairs, 90% of which was used for training set,
the remaining 10% is used for test set.

Preprocessing.

We used two tokenization schemes. The first is nltk's vanilla tokenization scheme, which includes:
	- lowercase all tokens
	- removing all non alphanumeric characters.

Note in the last case if a nonalphanumeric character appears inside of a word, then it is removed from the word. For example, w<rd becomes wrd. Punctuations are not removed.


Next we usedd tworkenize found in tworkenize.py, this include:
	- stripping white space
	- removing emojis that do not appear consecutively with no space in between
	- lower case
	- split off edge punctuation

See tworkenize.py for a comprehensive list of tokenization steps. 

Finally, we removed any tweet questions-response pairs where the question is longer than 20 tokens, and the question is shorter than 3 tokens or longer than 20 tokens. 

The vocabulary is limited to 6000 characters, all out of vocabulary (OOV) words are mapped to the token 'unk'.

Model.




Experiments.