# RolePlay-LLM
This repository contains the source code for the CS 521 class project (Spring 2024). The entire code is present as a single Jupyter notebook file. The code is adapted from these two amazing sources: this [script](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py) from Huggingface and great [tutorial](https://nathancooper.io/i-am-a-nerd/chatbot/deep-learning/gpt2/2020/05/12/chatbot-part-1.html) from Nathan Cooper. 

The code makes a decoder only LLM roleplay the neighbor and landlord for the neighbor and landlord scenarios respectively in Social Skills Performace Assessment (SSPA).

# Dataset
The data is based on the SSPA interviews of 2 class of patients: Bipolar Disorder and Schizophrenia. Since the data is sensitive, we will not release the dataset here.

# How to run
The notebook contains the code, so just running through the cells will work. The data will be needed to be uploaded to Gdrive for the code to work. Google colab subscription will be needed to run if GPU is being used. However, the following points need to be kept in mind:
1. There is a cell in the notebook titled: "Set the arguments here" The cell is used to set the training arguments. The following arguments need to be set:
   1. output_dir = Path to the directory that will hold the data related to the LLM currently being finetuned
   2. model_type = The model type
   3. model_name_or_path = The model id (from HuggingFace) of the model being finetuned or path to the folder containing model checkpoint
   4. config_name = Path to the LLM confoguration file
   5. tokenizer_name = Tokenizer id from HuggingFace or tokenizer checkpoint
   6. cache_dir = The path to cache directory
   7. BD_Scene1 = Path to the folder containing the data for Bipolar Scene 1
   8. self.BD_Scene2 = Path to the folder containing the data for Bipolar Scene 2
   9. self.HC_Scene1 = Path to the folder containing the data for Healthy Control Scene 1
   10. self.HC_Scene2 = Path to the folder containing the data for Healthy Control Scene 2
   11. self.SZ_Scene1 = Path to the folder containing the data for Schizophrenia Scene 1
   12. self.SZ_Scene2 = Path to the folder containing the data for Schizophrenia Scene 2
2. Running the code will generate model checkpoints during training. The no of steps after which the model is saved can be modified using the training argument save_steps. Make sure to set an appropriate value: A very low value will create a number of checkpoints, and a very large value will create very few checkpoints.
3. The code will also generate the result files in the results subfolder. The following result files will be generated:
   1. time_results.txt: This fill will contain the training time results
   2. evaluation_time.txt: This file will contain the evaluation time results
   3. evaluation_results.txt: This file will contain the evaluation result (BertScore and Rouge Score).
  
# Results
The results for the 3 models: DialoGPT-small, DialoGPT-medium, and TinyLlama-1.1B-chat are present in the results folder.

