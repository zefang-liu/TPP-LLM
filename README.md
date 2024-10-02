# TPP-LLM: Modeling Temporal Point Processes by Efficiently Fine-Tuning Large Language Models

This repository provides the implementation of **TPP-LLM**, a framework that integrates Temporal Point Processes (TPPs) with Large Language Models (LLMs) for event sequence prediction. The repository includes scripts for training and evaluating the TPP-LLM model on various real-world datasets.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/zefang-liu/TPP-LLM
   cd TPP-LLM
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add the source code to your Python path:
   ```bash
   export PYTHONPATH=$PYTHONPATH:<path-to-your-folder>/src
   ```

## Usage

To train and evaluate the model, use the provided configuration files. For example, to run the model on the Stack Overflow dataset:

```bash
python scripts/train_tpp_llm.py @configs/tpp_llm_so.config
```

You can adjust the configuration to run experiments on other datasets by selecting the appropriate config files located in the `configs/` directory.

## Datasets

The datasets used in this project can be downloaded and preprocessed by running the notebook `notebooks/tpp_data.ipynb`. This notebook will handle downloading, cleaning, and preparing the datasets for training. Supported datasets include:

- Stack Overflow
- Chicago Crime
- NYC Taxi
- U.S. Earthquake
- Amazon Reviews

Processed datasets will be stored in the `data/` directory.

## Citation

If you find this code useful in your research, please cite our paper:

```
```

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
