# RolePlay-LLM
This repository contains the source code for my MS project (Jan-July 2024). The entire code is present in the python file ms_project.py. The file MS project report presents the project report. The code is adapted from these two amazing sources: this [script](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py) from Huggingface and great [tutorial](https://nathancooper.io/i-am-a-nerd/chatbot/deep-learning/gpt2/2020/05/12/chatbot-part-1.html) from Nathan Cooper. 

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

# RolePlay-LLM: Fine-tuning Conversational LLMs for Structured Role-Based Dialogue

## Overview

This project focuses on fine-tuning large language models (LLMs) to generate **context-aware, role-consistent conversational dialogue** in structured scenarios. The primary use case explored is **clinical and behavioral roleplay conversations**, where maintaining context, speaker roles, and conversational coherence is critical.

Unlike generic chatbots, this system is designed to handle:
- Multi-turn dialogue with **strict role separation**
- Long conversational context
- Domain-specific nuances (e.g., clinical interviews)

---

## Motivation

Standard pretrained conversational models often:
- Lose **role consistency** across turns
- Fail to maintain **long-range context**
- Produce **generic or incoherent responses** in structured settings

This project aims to address these limitations by building a **fine-tuning pipeline that explicitly models conversational structure and constraints**.

---

## System Design

### 1. Data Processing Pipeline

- Converted raw transcripts into **structured multi-turn dialogue format**
- Explicitly encoded:
  - Speaker roles (e.g., interviewer vs subject)
  - Turn boundaries
- Handled noisy and inconsistent transcript formats through custom preprocessing

### 2. Training Strategy

- Used **Causal Language Modeling (CLM)** for dialogue generation
- Designed **selective loss masking**:
  - Model only learns to predict **target speaker responses**
  - Prevents leakage of context tokens into loss computation

### 3. Context Handling (Key Challenge)

**Problem:** Conversations exceed model token limits  

**Solution:**
- Implemented **sliding window context strategy**
- Preserved most recent and relevant turns
- Carefully balanced:
  - Context retention vs token budget
  - Information loss vs computational cost

---

## Key Technical Challenges

### 1. Long Context Truncation
- Naive truncation destroys conversational coherence
- Required designing **context-aware truncation logic**
- Tradeoff between:
  - Model performance
  - Memory constraints

---

### 2. Role Consistency

**Problem:** Model mixes speaker roles  

**Solution:**
- Introduced **explicit role tokens / prompt prefixes**
- Structured input formatting to reinforce role identity
- Iteratively refined prompt templates

---

### 3. Noisy Real-World Data

- Clinical transcripts contain:
  - Incomplete turns
  - Formatting inconsistencies
- Built robust preprocessing pipeline to:
  - Normalize dialogue
  - Remove artifacts
  - Maintain semantic integrity

---

### 4. Compute Constraints

- Fine-tuning large models is resource-intensive

**Optimizations:**
- **LoRA (Low-Rank Adaptation)** for parameter-efficient training
- Gradient checkpointing to reduce memory usage
- Mixed precision training

---

### 5. Evaluation Complexity

Traditional metrics are insufficient for dialogue.

**Approach:**
- Used **BERTScore** for semantic similarity
- Used **ROUGE** for lexical overlap
- Performed **qualitative evaluation**:
  - Coherence across turns
  - Role adherence
  - Context retention

---

## Tech Stack

- **Language**: Python  
- **Frameworks**: PyTorch, Hugging Face Transformers  
- **Training Utilities**: Accelerate, PEFT (LoRA)  
- **Evaluation**: BERTScore, ROUGE  
- **Data Processing**: Pandas, NumPy  

---

## Results

- Achieved **~0.8 BERTScore F1**, indicating strong semantic alignment
- Improved:
  - Context retention across turns
  - Role consistency in generated dialogue
- Generated outputs that are **qualitatively more structured and coherent** than baseline models

---

## Learnings

- Fine-tuning LLMs is less about architecture and more about:
  - **Data representation**
  - **Loss design**
  - **Context management**
- Evaluation remains a major bottleneck for conversational systems
- Small design decisions (prompt format, truncation strategy) have **outsized impact**

---

## Future Work

- Reinforcement learning for dialogue optimization
- Better evaluation frameworks (LLM-as-judge)
- Deployment as an interactive system
- Exploration of larger instruction-tuned models

---

## Conclusion

This project demonstrates how to move from a generic pretrained model to a **domain-adapted conversational system**, while navigating real-world constraints like noisy data, limited context windows, and compute limitations.
