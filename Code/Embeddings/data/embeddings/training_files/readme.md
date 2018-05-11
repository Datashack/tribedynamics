# Instructions
1) Select the folder of interest (en or it);

Once the "padded_dataset.txt" has been created, it can be split into multiple subfiles (each of the subfiles will correspond to one of the mini-batches to feed to the neural network).

2) Move the "padded_dataset.txt" inside the folder which will hold all the subfiles;

3) Run the following command from terminal: split -l BATCH-SIZE --numeric-suffixes --additional-suffix=.txt padded_dataset.txt padded_dataset_split
(BATCH-SIZE is the number of lines to include in each txt subfile; for example, for a batch_size of 32, we can run the following:
split -l 32 --numeric-suffixes --additional-suffix=.txt padded_dataset.txt padded_dataset_split)

4) Take out the "padded_dataset.txt" from the folder containing all the just created subfiles (we don't want to use the entire dataset together with the subfiles);

5) All the datafiles are ready to be loaded into the training script. Just make sure to provide to the parser of the training script the same number in the BATCH-SIZE parameter that was used to create the splits.

