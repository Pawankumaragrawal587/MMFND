# MMFND

environment.yml - conda environment for setup.

# Dataset

Currently Private

# Evaluation

Contains codes for running the results on the dataset presented in the paper.

As per requirements, Codes of all experiments are stored in 16 Subfolders inside Evaluation Folder. Each named from 1 to 16.
Following information can be used to understand which subfolder contains code for which experiment.

Folder               -               Experiment

1 		-	RoBERTa + VGG16 Based model on complete Dataset with Background Knowledge.

2		  -	RoBERTa + VGG16 Based model on ReCovery Dataset with Background Knowledge.

3		  -	RoBERTa + VGG16 Based model on Fauxtography Dataset with Background Knowledge.

4		  -	RoBERTa + VGG16 Based model on TICNN Dataset with Background Knowledge.

5		  -	RoBERTa + VGG16 Based model on complete Dataset without Background Knowledge.

6		  -	RoBERTa + VGG16 Based model on ReCovery Dataset without Background Knowledge.

7		  -	RoBERTa + VGG16 Based model on Fauxtography Dataset without Background Knowledge.

8		  -	RoBERTa + VGG16 Based model on TICNN Dataset without Background Knowledge.

9		  -	BERT + CNN Based model on complete Dataset with Background Knowledge.

10 		-	BERT + CNN Based model on ReCovery Dataset with Background Knowledge.

11 		-	BERT + CNN Based model on Fauxtography Dataset with Background Knowledge.

12		-	BERT + CNN Based model on TICNN Dataset with Background Knowledge.

13		-	BERT + CNN Based model on complete Dataset without Background Knowledge.

14		-	BERT + CNN Based model on ReCovery Dataset without Background Knowledge.

15		-	BERT + CNN Based model on Fauxtography Dataset without Background Knowledge.

16 		-	BERT + CNN Based model on TICNN Dataset without Background Knowledge.

Each subfolder contains a TrainSiameseNet.py file. Run this file to start the training of corresponding model.


