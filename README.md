# Data Discovery and Probabilistic Relationship Analysis

This project implements a sophisticated system for discovering and analyzing probabilistic relationships between entities and their attributes in text data. It uses Markov chains and probabilistic graphs to model and visualize relationships between different entities and their attributes.

## Features

- **Probabilistic Relationship Modeling**: Builds conditional probability graphs between entities and their attributes
- **Natural Language Processing**: Uses GPT-4 to extract entity-attribute relationships from text
- **Visualization**: Generates interactive visualizations of relationship graphs
- **Flexible Schema Support**: Works with custom JSON schemas to define entity-attribute relationships
- **Bidirectional Probability Analysis**: Computes both forward and backward conditional probabilities

## Project Structure

- `markovChain.py`: Core implementation of the Markov chain and probabilistic graph
- `probGen.py`: Main script for generating probability graphs from text data
- `update_probGen.py`: Enhanced version with additional features
- Various JSON files for schemas and generated chains
- Visualization outputs in PNG format

## Requirements

- Python 3.6+
- Required packages:
  - numpy
  - pandas
  - networkx
  - matplotlib
  - openai

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key'
```

## Usage

1. Prepare your input data in CSV format
2. Define your schema in JSON format
3. Run the main script:
```bash
python probGen.py
```

## Output

The system generates:
- A filled schema JSON file
- A Markov chain JSON file
- Visualizations of the relationship graphs

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 