1. Step 1: Create vocabulary tokens and then choosing good lexicons. Good lexicons can be downloaded or created manually. utils.py has all
key functions related to getting good translations. get_topn_translations() uses python's vocabulary package and gets the top translations

2. Step 2 : Run EM(Expectation Maximization) using em_train.py

3. Step 3: Use lexicon and the EM probabilities in classify.py

The exact procedure can be found in http://web.eecs.umich.edu/~mihalcea/papers/shi.emnlp10.pdf
