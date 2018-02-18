## SOFTWARE DESIGN INSTRUCTIONS

1. main.py - Entry script will be  . It will be setup in such a way that hyperparams and different settings can be sent through command line 
and we invoke different models via the same argparser from command line.

2. data_process.py - This script will be used for all data processing stuff, such as conversion of data into vectors, bigrams etc. to be 
directly presented to model.

3. models.py - Each different model should be declared as a class. Let us follow this architecture even if it is a one line model

4. train.py - Script that invokes different model classes and trains them based on data obrained from data_process and model 
from the command line arguments. Will either save the model or directly invoke predict.py

5. predict.py - this will be a script that will load either a pre-trained model or will get in a model from train.py and performs
validation/test set analysis and reports metrics.

5. model_eval_metrics.py - All the model evaluation script code will reside in here.

6. utils.py - Different utils script will reside here.

7. const.py - All important constant values will reside HERE and HERE ONLY. 


