# SOC-Project- A Voyage into LLMs: From Inception to Mastery - Midterm Report

**Student:** Pankaj Gurjar  
**Department:** Chemical Engineering  
**Roll No.:**  24b0328                                                                                                                             
**Project:** Summer of Code (SOC) - A Voyage into LLMs: From Inception to Mastery

## 📋 Project Overview

This repository documents my learning journey through the first half of an 8-week Summer of Code project focused on Large Language Models (LLMs) and Natural Language Processing (NLP). The project aims to provide a comprehensive understanding of LLMs and NLP fundamentals, starting from basic Python programming and progressing through neural networks to advanced language models.

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

## 📁 Repository Structure

```
├── README.md                 # This comprehensive project documentation
├── Assignment.py             # MNIST digit recognition implementation
├── Week1.py                  # NumPy and Pandas fundamentals
├── Week2.py                  # Text processing and NLP concepts
├── Week3-4.py                # Neural network implementation examples
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

## 🔄 Next Steps (Weeks 5-8)

### Upcoming Topics
- **Week 5:** Attention mechanisms and their applications
- **Week 6:** Transformer architecture and Hugging Face integration
- **Week 7:** Fine-tuning pre-trained models, RAG, and LangChain
- **Week 8:** Final project implementation and presentation

### Planned Enhancements
- Implement sequence-to-sequence models
- Explore pre-trained language models
- Build practical NLP applications
- Develop understanding of modern LLM architectures



