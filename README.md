# SOC-Project- A Voyage into LLMs: From Inception to Mastery - Midterm Report

**Student:** Pankaj Gurjar  
**Department:** Chemical Engineering  
**Roll No.:**  24b0328                                                                                                                             
**Project:** Summer of Code (SOC) - A Voyage into LLMs: From Inception to Mastery

## 📋 Project Overview
This repository records my 8-week deep dive, moving from Python and basic data science into the architecture, theory, and code of Large Language Models (LLMs). The capstone achievement is a GPT-inspired autoregressive language model, coded entirely from scratch—demonstrating the inner workings powering modern generative AI.


## 🎯 Project Objectives

- Master foundational Python libraries (NumPy, Pandas, PyTorch)
- Understand text processing and NLP fundamentals
- Implement neural networks from scratch and using frameworks
- Explore sequence models (RNNs, LSTMs, GRUs)
- Progress towards understanding attention mechanisms and transformers
- Build practical skills in machine learning and deep learning

## 📚 Learning Progress (Weeks 1-4)

### Week 1: Python Foundations and Essential Libraries
**Topics Covered:**
- Python refresher and programming fundamentals
- NumPy for numerical computing and array operations
- Pandas for data manipulation and analysis
- Matplotlib for data visualization
- PyTorch basics for deep learning

**Key Skills Acquired:**
- Creating and manipulating NumPy arrays
- Data loading, cleaning, and preprocessing with Pandas
- Basic visualization techniques
- Understanding PyTorch tensors and basic operations

### Week 2: Natural Language Processing Fundamentals
**Topics Covered:**
- Regular expressions for pattern matching
- Text preprocessing techniques (tokenization, stemming, lemmatization)
- Word embeddings and vectorization methods (TF-IDF, Word2Vec)
- Sentiment analysis fundamentals
- NLTK and spaCy libraries for NLP tasks

**Key Skills Acquired:**
- Text cleaning and preprocessing pipelines
- Feature extraction from text data
- Understanding of different text representation methods
- Basic sentiment analysis implementation

### Week 3-4: Neural Networks and Deep Learning
**Topics Covered:**
- Artificial neural network fundamentals
- Forward propagation and backpropagation
- Loss functions and optimization algorithms
- Introduction to Recurrent Neural Networks (RNNs)
- PyTorch model implementation

**Key Skills Acquired:**
- Building neural networks from scratch
- Understanding gradient descent and optimization
- Implementing feedforward networks using PyTorch
- Model training, validation, and testing procedures

## 🚀 Key Assignment: MNIST Digit Recognition

### Project Description
Implemented a comprehensive digit recognition system using PyTorch to classify handwritten digits from the MNIST dataset. This project served as a practical application of neural network concepts learned during the program.

### Technical Implementation
- **Architecture:** Three-layer feedforward neural network
- **Input Layer:** 784 neurons (28×28 flattened images)
- **Hidden Layers:** 256 and 128 neurons with ReLU activation
- **Output Layer:** 10 neurons for digit classification (0-9)
- **Training:** Adam optimizer with Cross-Entropy loss
- **Performance:** Achieved ~97-98% accuracy on test set

### Key Features
- Proper train/validation/test split 
- Comprehensive model evaluation and metrics
- Visualization of predictions with correctness indicators
- Well-documented code with detailed comments

Week 5: Attention Mechanisms and Visualization
Topics Covered:
Concept of attention in sequence models
Additive vs. multiplicative (dot-product) attention
Scaled dot-product attention formula
Visualizing attention weights on sample sentences

Key Skills Acquired:
Implementing attention layers from first principles
Visualizing alignment matrices for interpretability
Integrating attention into RNN-based sequence-to-sequence models

Week 6: Transformer Architecture and Hugging Face Integration
Topics Covered:
Architecture of the Transformer (encoder and decoder blocks)
Multi-head attention, positional encodings
Layer normalization and residual connections
Introduction to Hugging Face transformers library

Key Skills Acquired:
Building Transformer encoder and decoder layers in PyTorch
Loading pre-trained models (e.g., bert-base-uncased, gpt2)
Tokenization and pipeline APIs for inference

Week 7: Fine-Tuning, RAG, and LangChain
Topics Covered:
Fine-tuning pre-trained LLMs on custom datasets
Retrieval-Augmented Generation (RAG) concepts
Building simple document retrievers with FAISS
Introduction to LangChain for chaining LLM calls

Key Skills Acquired:
Designing training loops to fine-tune GPT-2 on dialogue data
Implementing a FAISS index for efficient document retrieval
Composing LangChain chains (LLM + retriever + prompt template)

Week 8: Final Project Implementation and Presentation
Topics Covered:
End-to-end GPT-from-scratch design
Training custom tokenizer and vocabulary
Transformer training loops and checkpointing
Model evaluation and deployment considerations

Key Skills Acquired:
Implementing Byte-Pair Encoding (BPE) tokenizer
Writing efficient data loaders and batching strategies
Monitoring training with TensorBoard
Exporting and serving models as REST APIs
Capstone Project: "Build a GPT from Scratch"
Completed architecture implementation through Week 8
Trained on a sample corpus of programming Q&A
Achieved coherent code-completion demos


## 📁 Repository Structure

```
├── README.md                 # This comprehensive project documentation
├── Assignment.py             # MNIST digit recognition implementation
├── Week1.py                  # NumPy and Pandas fundamentals
├── Week2.py                  # Text processing and NLP concepts
├── Week3-4.py                # Neural network implementation examples
├── Week5                     # Attention mechanisms and visualizations
├── Week6                     # Transformer layers and Hugging Face integration
├── Week7                     # RAG and LangChain prototypes
├── Week8                     # GPT-from-scratch implementation
```

## 📦 File Descriptions

### `Assignment.py`
Complete implementation of the MNIST digit recognition project including:
- Data loading and preprocessing
- Model architecture definition
- Training and validation loops
- Testing and performance evaluation
- Visualization of results

### `Week1.py`
Practical examples demonstrating:
- NumPy array operations and mathematical computations
- Pandas DataFrame manipulation and data analysis
- Basic data visualization with Matplotlib

### `Week2.py`
Implementation of NLP fundamentals:
- Text preprocessing pipelines
- Tokenization and text cleaning
- TF-IDF vectorization
- Basic sentiment analysis

### `Week3-4.py`
Neural network concepts and implementations:
- Simple perceptron implementation
- Basic feedforward network
- Gradient descent demonstration
### 'SOC Final Project - GPT from Scratch.py'
Complete GPT-style language model built from scratch:
Tokenizer training with Byte-Pair Encoding (BPE)
Positional encoding and multi-head self-attention
Transformer blocks with residual connections
Causal masking for autoregressive text generation
Custom dataset loading, batching, and training loop
Final evaluation and inference interface


## 🎓 Learning Outcomes

### Technical Skills Developed
- **Data Manipulation:** Proficient in NumPy arrays and Pandas DataFrames
- **Text Processing:** Understanding of NLP preprocessing pipelines
- **Deep Learning:** Ability to implement and train neural networks
- **Model Evaluation:** Knowledge of performance metrics and validation techniques
- **Code Organization:** Writing clean, documented, and modular code

### Conceptual Understanding
- Mathematical foundations of neural networks
- Text representation and feature extraction methods
- Training procedures and optimization algorithms
- Model architecture design principles
- Overfitting, underfitting, and regularization concepts

## 🔬 Experimental Results

### MNIST Digit Recognition Performance
- **Training Accuracy:** Consistently reached 99%+ by epoch 10
- **Validation Accuracy:** Stable at 97-98% throughout training
- **Test Accuracy:** Final performance of 97.8% ± 0.2%
- **Training Time:** ~2-3 minutes on CPU, <1 minute on GPU

### Key Observations
- Model converged quickly within first 5 epochs
- No significant overfitting observed with current architecture
- ReLU activation proved effective for this classification task
- Adam optimizer outperformed SGD in convergence speed





