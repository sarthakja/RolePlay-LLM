# RolePlay-LLM
This repository contains the source code for my MS project (Jan-July 2024). The entire code is present in the python file ms_project.py. The file MS_project_report presents the project report. The code is adapted from these two amazing sources: this [script](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py) from Huggingface and great [tutorial](https://nathancooper.io/i-am-a-nerd/chatbot/deep-learning/gpt2/2020/05/12/chatbot-part-1.html) from Nathan Cooper. 

The code makes different LLMs (DialoGPT, TinyLlama, Llama2-chat, T5) roleplay the neighbor and landlord for the neighbor and landlord scenarios respectively in Social Skills Performace Assessment (SSPA). The report of the project is present in the file CS_521_final_report.

# Dataset
The data is based on the SSPA interviews of 2 class of patients: Bipolar Disorder and Schizophrenia. Since the data is sensitive, we will not release the dataset here.

# How to run
The code file uses the [accelerate framework](https://huggingface.co/docs/accelerate/en/index) from huggingface to run the code on multiple GPUs. The code can also be configured to utilize LoRA framework for parameter efficient finetuning.  There is a class called Args that is used to configure the hyperparameters and other settings of the code. The first point below presents the details of some of them.
1. The description of some of the variables in the Args class is as follows:
   1. output_dir (String) = Path to the directory that will hold the data related to the LLM currently being finetuned
   2. model_type (String) = The model type
   3. model_name_or_path (String) = The model id (from HuggingFace) of the model being finetuned or path to the folder containing model checkpoint
   4. config_name (String) = Path to the LLM configuration file
   5. tokenizer_name (String) = Tokenizer id from HuggingFace or tokenizer checkpoint
   6. cache_dir (String) = The path to cache directory
   7. BD_Scene1 (String) = Path to the folder containing the data for Bipolar Scene 1
   8. self.BD_Scene2 (String) = Path to the folder containing the data for Bipolar Scene 2
   9. self.HC_Scene1 (String) = Path to the folder containing the data for Healthy Control Scene 1
   10. self.HC_Scene2 (String) = Path to the folder containing the data for Healthy Control Scene 2
   11. self.SZ_Scene1 (String) = Path to the folder containing the data for Schizophrenia Scene 1
   12. self.SZ_Scene2 (String) = Path to the folder containing the data for Schizophrenia Scene 2
   13. self.test_size (float) = Percentage of examples in the test set
   14. self.context_size (integer) = The context size of LLM
   15. self.new_tokens (integer) = Number of tokens to be generated
   16. self.create_dataframe (Boolean) = Whether or not to create the train/test split of the data. It is recommended to set this to true once initially to create the split, and then use the same split for all other models.
   17. self.should_perform_lora_training (Boolean) = Whether or not to perform LoRA training. It is recommended to set this to True for larger models that cannot be trained on a single GPU.
   18. self.should_perform_gradient_checkpointing (Boolean) = Whether or not to perform gradient checkpointing during training. This is useful when it is difficult to train the model on a single GPU even after using LoRA
   19. self.lora_rank (integer) = The rank of the the decomposed matrices in LoRA.
   20.  self.generate_transcript (Boolean) = Whether or not to generate the transcript for the test set examples. If set to True, the transcripts are present at the path results/model_generated_output
   21.  self.promptStringNeighbor (String) = The string used as a prompt for the neighbor scenario
   22.  self.promptStringLandlord (String) = The string used as a prompt for the landlord scenario  
2. Running the code will generate model checkpoints during training. The no of steps after which the model is saved can be modified using the training argument save_steps. Make sure to set an appropriate value: A very low value will create a number of checkpoints, and a very large value will create very few checkpoints.
3. The code will also generate the result files in the results subfolder. The following result files and folder will be generated:
   1. time_results.txt: This fill will contain the training time results
   2. evaluation_time.txt: This file will contain the evaluation time results
   3. evaluation_results.txt: This file will contain the evaluation result (BertScore and Rouge Score).
   4. model_generated_output: This folder contains the transcripts generated for test set examples. Each subfolder contains transcripts for a particular scene.
  
# Results
The results for the all the models are present in the results subfolder.
