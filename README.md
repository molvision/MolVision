# ChemVision Inference Tool

A Python tool for running various vision language model inference on chemical datasets from HuggingFace, specifically designed for molecular image analysis and chemical property prediction tasks.

## 🔗 Related Links

- **Project Website**: [ChemVision](https://chemvision.github.io/chemvision/)
- **HuggingFace Organization**: [ChemVision on HuggingFace](https://huggingface.co/ChemVision)


## 📋 Introduction

Molecular property prediction is a fundamental task in computational chemistry with critical applications in drug discovery and materials science. While recent works have explored Large Language Models (LLMs) for this task, they primarily rely on textual molecular representations such as SMILES/SELFIES, which can be ambiguous and structurally less informative. In this work, we introduce MolVision, a novel approach that leverages Vision-Language Models (VLMs) by integrating both molecular structure as images and textual descriptions to enhance property prediction. We construct a benchmark spanning ten diverse datasets, covering classification, regression and description tasks. Evaluating nine different VLMs in zero-shot, few-shot, and fine-tuned settings, we find that visual information improves prediction performance, particularly when combined with efficient fine-tuning strategies such as LoRA. Our results reveal that while visual information alone is insufficient, multimodal fusion significantly enhances generalization across molecular properties. Adaptation of vision encoder for molecular images in conjunction with LoRA further improves the performance.

## 🧬 MolVision - Characteristics and Statistics

### Characteristics of MolVision

- **Multimodal Integration**: MolVision combines skeletal structure images with SMILES representations for molecular property prediction
- **Diverse Datasets**: It includes Ten datasets covering various molecular properties and complexities
- **Evaluation Scenarios**: Assessing Vision-Language Models (VLMs) under zero-shot, few-shot, Chain-of-Thought and fine-tuning conditions
- **Comparative Analysis**: Benchmarking Two closed source and Seven Opensourced different VLMs to analyze their effectiveness in computational chemistry

### Statistics of MolVision

| Category | Details |
|----------|---------|
| Number of Datasets | 10 datasets: BACE-V, BBBP-V, HIV-V, Clintox-V, Tox21-V, Esol-V, LD50-V, QM9-V, PCQM4Mv2-V, Chebi-V |
| Dataset Composition | Includes skeletal structure images and corresponding SMILES strings |
| Model Evaluation | Two Closed Source and Seven OpenSourced Vision-Language Models evaluated |
| Performance Metrics | Measured across zero-shot, few-shot, Chain-of-thought and fine-tuning scenarios |

## 📋 Overview

This tool enables automated inference using OpenAI's GPT-4 vision models on chemical datasets containing molecular images and associated questions/prompts. It's particularly useful for:

- Molecular property prediction
- Chemical structure analysis
- Multi-modal AI evaluation on chemistry datasets
- Batch processing of vision-language tasks in chemistry

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) for environment management
- OpenAI API key

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/chemvision/chemvision.git
   cd chemvision
   ```

2. **Create and activate virtual environment with uv**:
   ```bash
   uv sync
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Set up OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## 📖 Usage

### Basic Usage

Run inference on the default ChemVision dataset:

```bash
python main.py
```

### Advanced Usage

```bash
python main.py \
    --dataset "ChemVision/BACE-V-SMILES-2" \
    --split "train" \
    --model "gpt-4o" \
    --output "results.csv" \
    --num_samples 50 \
    --save_interval 10
```



## 📊 Output Format

The tool generates a CSV file with the following columns:

- `sample_id`: Index of the processed sample
- `question`: The input question/prompt
- `image_path`: Path to the associated molecular image
- `has_image`: Boolean indicating if image was successfully processed
- `gpt_response`: The model's response/prediction

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Supported Dataset Formats

The tool automatically detects and handles:
- CSV metadata files
- JSONL metadata files
- PNG/JPG molecular images
- Various column naming conventions for questions/prompts

