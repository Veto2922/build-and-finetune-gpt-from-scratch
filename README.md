# GPT from Scratch

A comprehensive educational project that implements a GPT (Generative Pre-trained Transformer) model from scratch using PyTorch. This project covers everything from data preprocessing and tokenization to building transformer blocks, training, and fine-tuning for various tasks.

## 📚 Project Overview

This project is a complete implementation of GPT architecture, built step-by-step through a series of Jupyter notebooks. It demonstrates:

- **Data Processing**: Cleaning and preprocessing Arabic text data (Sahih al-Bukhari)
- **Tokenization**: Building a custom Unigram tokenizer for Arabic text
- **Model Architecture**: Implementing all GPT components from scratch:
  - Embeddings (token and positional)
  - Multi-head self-attention mechanism
  - Layer normalization
  - Feed-forward networks
  - Transformer blocks
  - Complete GPT model
- **Training**: Training loops, evaluation, and optimization
- **Decoding Strategies**: Implementing various text generation strategies
- **Fine-tuning**: 
  - Classification fine-tuning (SMS spam detection)
  - Instruction fine-tuning for task-specific responses
- **Model Loading**: Loading and using OpenAI's pretrained GPT-2 weights

## 🏗️ Project Structure

```
GPT_from_scratch/
├── notebooks/                    # Main work directory - step-by-step implementation
│   ├── 01_data_cleaning.ipynb
│   ├── 02_Tokenization.ipynb
│   ├── 03_data_loader.ipynb
│   ├── 04_embedding.ipynb
│   ├── 05_attention.ipynb
│   ├── 06_dummy_gpt_model.ipynb
│   ├── 07_layer_norm.ipynb
│   ├── 08_feed_forward.ipynb
│   ├── 09_transformer_block.ipynb
│   ├── 10_GPT_MODEL.ipynb
│   ├── 11_training_and_eval_loops.ipynb
│   ├── 12_Decoding_stratefies.ipynb
│   ├── 13_loading_openAi_pretrained_weights.ipynb
│   ├── fine-tuning_for_classification/
│   │   └── 01_classification_fine-tuning.ipynb
│   └── Instruction_Finetuning/
│       ├── instruction_fine-tuning.ipynb
│       └── model_evaluation.ipynb
├── src/                          # Production-ready source code
│   ├── GPT_model.py             # Complete GPT model implementation
│   ├── GPT_blocks/               # Modular GPT components
│   │   ├── attention_blocks.py
│   │   ├── feed_forward.py
│   │   ├── layer_norm.py
│   │   └── transformer_block.py
│   └── utils/                    # Utility functions
│       ├── data_loader.py
│       ├── generate_text_simple.py
│       └── gpt_download.py
├── data/                         # Dataset directory (not in repo)
│   ├── Sahih_al-Bukhari.txt
│   └── cleaned_text.txt
├── tokenizer_model/              # Trained tokenizer models (not in repo)
│   ├── hadith_unigram.model
│   └── hadith_unigram.vocab
├── pyproject.toml                # Project dependencies
└── README.md                     # This file
```

## 📖 Notebooks Walkthrough

### Core Implementation Notebooks

#### 1. **01_data_cleaning.ipynb**
- Loads Arabic text data (Sahih al-Bukhari)
- Implements text cleaning pipeline:
  - Removes diacritics (tashkeel)
  - Removes brackets and annotations
  - Removes numbers and punctuation
  - Normalizes whitespace
- Outputs cleaned text for tokenization

#### 2. **02_Tokenization.ipynb**
- Implements Unigram tokenizer using SentencePiece
- Trains tokenizer on Arabic text
- Creates vocabulary of 22,000 tokens
- Saves tokenizer model and vocabulary files

#### 3. **03_data_loader.ipynb**
- Creates PyTorch Dataset class for GPT training
- Implements sliding window approach for sequence chunking
- Creates DataLoader with batching and shuffling

#### 4. **04_embedding.ipynb**
- Implements token embeddings
- Implements positional embeddings
- Combines embeddings for transformer input

#### 5. **05_attention.ipynb**
- Explains self-attention mechanism from first principles
- Demonstrates query, key, value computation
- Shows attention score calculation
- Implements scaled dot-product attention

#### 6. **06_dummy_gpt_model.ipynb**
- Creates a simplified GPT model for testing
- Tests basic forward pass
- Validates model architecture

#### 7. **07_layer_norm.ipynb**
- Implements Layer Normalization
- Explains normalization techniques
- Tests layer norm on sample data

#### 8. **08_feed_forward.ipynb**
- Implements feed-forward network (FFN)
- Two linear layers with GELU activation
- Dropout for regularization

#### 9. **09_transformer_block.ipynb**
- Combines all components into Transformer Block:
  - Multi-head self-attention
  - Layer normalization
  - Feed-forward network
  - Residual connections
- Implements causal masking for autoregressive generation

#### 10. **10_GPT_MODEL.ipynb**
- Implements complete GPT model architecture
- Combines:
  - Token and positional embeddings
  - Stack of transformer blocks
  - Final layer normalization
  - Output head for vocabulary prediction
- Configures GPT-2 124M parameter model

#### 11. **11_training_and_eval_loops.ipynb**
- Implements training loop with:
  - Forward pass
  - Loss calculation (cross-entropy)
  - Backward pass and optimization
  - Learning rate scheduling
- Implements evaluation loop
- Tracks training metrics
- Saves model checkpoints

#### 12. **12_Decoding_stratefies.ipynb**
- Implements text generation strategies:
  - Greedy decoding
  - Top-k sampling
  - Top-p (nucleus) sampling
  - Temperature scaling
- Demonstrates different generation behaviors

#### 13. **13_loading_openAi_pretrained_weights.ipynb**
- Downloads OpenAI's GPT-2 124M model weights
- Converts TensorFlow checkpoints to PyTorch
- Loads pretrained weights into custom model
- Tests pretrained model for text generation

### Fine-tuning Notebooks

#### **fine-tuning_for_classification/01_classification_fine-tuning.ipynb**
- Fine-tunes GPT for SMS spam classification
- Downloads SMS Spam Collection dataset
- Creates balanced dataset
- Adds classification head to GPT
- Trains on classification task
- Evaluates model performance

#### **Instruction_Finetuning/instruction_fine-tuning.ipynb**
- Implements supervised fine-tuning (SFT) for instruction following
- Uses instruction-response dataset
- Formats data with instruction/input/response structure
- Implements custom collate function for variable-length sequences
- Trains model to follow instructions
- Evaluates instruction-following capabilities

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- PyTorch 2.3.0 or higher
- CUDA-capable GPU (recommended for training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Veto2922/build-and-finetune-gpt-from-scratch.git
cd GPT_from_scratch
```

2. Install dependencies using `uv` (recommended) :
```bash
# Using uv
uv sync
```

3. Prepare your data:
   - Place your text data in the `data/` directory
   - Or use the provided data cleaning notebook to process your own dataset

### Running the Notebooks

1. Start Jupyter Lab:
```bash
jupyter lab
```

2. Navigate through the notebooks in order (01-13) to understand the complete implementation

3. For fine-tuning tasks:
   - Classification: `notebooks/fine-tuning_for_classification/01_classification_fine-tuning.ipynb`
   - Instruction following: `notebooks/Instruction_Finetuning/instruction_fine-tuning.ipynb`

## 🔧 Key Features

### Model Architecture
- **GPT-2 124M Configuration**:
  - Vocabulary size: 50,257 (GPT-2 tokenizer)
  - Context length: 1024 tokens
  - Embedding dimension: 768
  - Attention heads: 12
  - Transformer layers: 12
  - Dropout: 0.1

### Custom Tokenizer
- Unigram tokenizer trained on Arabic text
- Vocabulary size: 22,000 tokens
- Optimized for morphologically rich languages

### Training Features
- Sliding window data loading
- Gradient accumulation support
- Learning rate scheduling
- Model checkpointing
- Training metrics tracking

### Generation Features
- Multiple decoding strategies
- Configurable temperature and sampling parameters
- Causal masking for autoregressive generation

## 📊 Datasets Used

1. **Sahih al-Bukhari**: Arabic hadith collection for pretraining
2. **SMS Spam Collection**: For classification fine-tuning
3. **Instruction Dataset**: For instruction-following fine-tuning

## 🎯 Use Cases

This implementation can be used for:
- **Text Generation**: Generate coherent text in Arabic or English
- **Classification**: Fine-tune for text classification tasks
- **Instruction Following**: Create models that follow specific instructions
- **Educational Purposes**: Learn transformer architecture from the ground up

## 📝 Code Organization

The project follows a modular structure:

- **Notebooks**: Educational, step-by-step implementation with explanations
- **src/**: Production-ready, reusable code modules
- **GPT_blocks/**: Individual components (attention, FFN, layer norm, etc.)
- **utils/**: Helper functions for data loading, generation, and model downloading

## 🔬 Technical Highlights

1. **From Scratch Implementation**: Every component is built from basic PyTorch operations
2. **Educational Focus**: Extensive comments and explanations in notebooks
3. **Modular Design**: Components can be reused and extended
4. **Best Practices**: Follows PyTorch and Python best practices
5. **Comprehensive Coverage**: From data preprocessing to advanced fine-tuning

## 📚 Learning Path

Recommended order for learning:

1. **Basics** (Notebooks 01-04): Data processing and embeddings
2. **Core Components** (Notebooks 05-09): Attention, normalization, and transformer blocks
3. **Complete Model** (Notebooks 10-11): Full GPT model and training
4. **Advanced Topics** (Notebooks 12-13): Generation strategies and pretrained models
5. **Fine-tuning** (Fine-tuning notebooks): Task-specific adaptations

## 🤝 Contributing

This is an educational project. Feel free to:
- Report issues
- Suggest improvements
- Add more examples
- Extend functionality

## 📄 License

This project is for educational purposes. Please respect the licenses of:
- OpenAI GPT-2 (if using pretrained weights)
- Any datasets you use

## 🙏 Acknowledgments

- OpenAI for the GPT architecture and pretrained models
- The PyTorch community
- SentencePiece for tokenization
- All contributors to the open-source ML ecosystem

## 📧 Data Source

The Arabic text data used in this project is sourced from:
https://huggingface.co/spaces/ieasybooks-org/ShamelaWaqfeya

---

**Note**: This project is designed for educational purposes to understand transformer architectures and GPT models. For production use, consider using established libraries like Hugging Face Transformers.
