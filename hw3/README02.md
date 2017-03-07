### MODELS ###
We used vanilla sequence to sequence model with attention mechanism, first proposed by Cho et al. (https://arxiv.org/pdf/1406.1078.pdf). This model was originally designed for machine translation and is trained to maximize the probabilty of target sequence given input sequence, where the cost is cross entropy. The model maps in the input sequence into a hidden vector, where the attention mechanism controls how much hidden information will propogate forward.

We utilized the orginial "translate.py" in the tensorflow repository written for French-English translation. All functions in the original file have been slightly modified for our twitter chatbot. In total, we trained three different models as follow:
	- preprocessed with nltk's vanilla tokenization scheme
	- preprocessed with tworkenize.py
	- preprocessed with tworkenize.py & early stopping
The total vocabulary size for both questions and answers is 6004 including the default special vocabulary used in Seq2Seq model (_PAD_, _UNK_, _GO_, _EOS_). We used four buckets with the following bucket sizes (question, answer)-pairs: [(5, 5), (10, 10), (15, 15), (25, 25)].

### TEST RESULTS and DISCUSSION ###
We tested each model with the same set of sentences including some sentences in the training data. The results are in the "results" folder. Detailed explanantions are denoted below.

1) Preprocessed with NLTK's Vanilla Tokenization scheme
- learning rate: 0.1
- The total number of training steps: 107600
- Perplexity at the stopping point: 
	- Global: 27.09
	- Buckets: 122.35, 72.53, 70.51, 87.10
- Reuslt files:
	"nltk_107600.txt"
	"nltk_107600.png" (Just screenshot image)

2) Preprocessed with tworkenize.py
- learning rate: 0.5
- The total number of training steps: 128200
- Perplexity at the stopping point: 
	- Global: 1.11
	- Buckets: 2627.52, 16880.32, 43720.34, 24863.28
- Reuslt files:
	"tworken_128200.txt"
	"tworken_128200.png" (Just screenshot image)

3) Preprocessed with tworkenize.py & early stopping
- learning rate: 0.5
- The total number of training steps: 27000
- Perplexity at the stopping point:
	- Global: 25.12
	- Buckets: 49.00, 48.75, 79.21, 92.70
- Reuslt files:
	"tworken_earlystopping_27000.txt"
	"tworken_earlystopping_27000.png" (Just screenshot image)





