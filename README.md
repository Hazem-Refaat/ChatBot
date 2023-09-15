# ChatBot
ChatBot using GBT/Dialogbt

**1. Loading a Custom Dataset**
   - You load a custom dataset using the Hugging Face `datasets` library. 

**2. Data Preprocessing**
   - You define a function `concatenate_utterances` to concatenate user queries and assistant responses into a single "dialog" string.

**3. Tokenization**
   - You use the Hugging Face `AutoTokenizer` to load a pre-trained tokenizer (`microsoft/DialoGPT-medium`) for tokenizing text.
   - The tokenizer's padding token is set to the end-of-sequence token.
   
**4. Model Loading**
   - You load a pre-trained DialoGPT model (`microsoft/DialoGPT-medium`) using `AutoModelForCausalLM`.

**5. Data Preparation for Training**
   - You encode and prepare the training data by tokenizing and creating tensors.
   - The data is organized into a PyTorch `TensorDataset` and then into a `DataLoader` for efficient batching.

**6. Multi-GPU Training**
   - You define a list of GPU device IDs and move the model to the GPU using DataParallel.

**7. Loss and Optimizer**
   - You define a cross-entropy loss (`nn.CrossEntropyLoss`) and an AdamW optimizer for training the model.

**8. Training Loop**
   - You train the model for a specified number of epochs.
   - For each epoch, you iterate through the training data, calculate the loss, perform backpropagation, and update the model's weights.
   - You print the average loss for each epoch.

**9. Model and Tokenizer Saving**
   - After training, you save the trained model and tokenizer for later use.

**10. Inference Loop**
   - You set up a loop to interact with the trained model.
   - You take user input, tokenize it, and generate a response from the model.
   - The model response is decoded and printed as the bot's reply.
   - The loop continues until the user enters "exit."

Overall, this code accomplishes the following tasks:
- Loads a custom dataset for training a dialogue model.
- Preprocesses the data, tokenizes it, and prepares it for training.
- Trains the model using cross-entropy loss and multi-GPU training.
- Saves the trained model and tokenizer.
- Sets up an interactive loop for users to have conversations with the trained chatbot model.

Please note that this code is designed for a specific use case of training a chatbot model and may require additional adjustments or customizations for other tasks or datasets.
