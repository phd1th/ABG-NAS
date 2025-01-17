# ABG-NAS
Code repository for ABG-NAS framework 
# ABG-NAS: Adaptive Bayesian Genetic Neural Architecture Search

ABG-NAS (Adaptive Bayesian Genetic Neural Architecture Search) is an innovative framework designed to optimize Graph Neural Networks (GNNs) through efficient architecture search and dynamic hyperparameter tuning. By leveraging advanced genetic algorithms and Bayesian optimization, ABG-NAS achieves exceptional performance across diverse graph datasets.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Datasets](#datasets)
- [Experimental Results](#experimental-results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

ABG-NAS introduces a novel framework for automated GNN architecture search. It incorporates three core components to balance exploration, exploitation, and scalability:

### Core Components
- **Comprehensive Architecture Search Space (CASS):** Allows diverse combinations of propagation and transformation operations, enabling the discovery of robust GNN architectures.
- **Adaptive Genetic Optimization Strategy (AGOS):** Dynamically adjusts genetic algorithm strategies, enhancing search efficiency across different graph structures.
- **Bayesian-Guided Tuning Module (BGTM):** Periodically optimizes hyperparameters, ensuring high performance while minimizing computational overhead.

### Applications
ABG-NAS is effective for tasks like node classification, link prediction, and subgraph search, especially in sparse and dense graph environments.

---

## Features
- **Efficient GNN Search:** Identifies high-performing GNN architectures in complex search spaces.
- **Dynamic Adaptation:** AGOS adjusts its parameters based on evolutionary stages, optimizing search processes.
- **Phased Hyperparameter Optimization:** BGTM tunes key hyperparameters at regular intervals for optimal performance.
- **Robust Cross-Dataset Performance:** Demonstrates superior results across diverse datasets (e.g., Cora, PubMed, Citeseer).

---

## Installation

To install and run ABG-NAS, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/sserranw/ABG-NAS.git
   cd ABG-NAS
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment (optional):
   ```bash
   conda create -n abg-nas python=3.8
   conda activate abg-nas
   ```
## Datasets
ABG-NAS has been tested on the following datasets:
- **Cora**
- **PubMed**
- **CoraFull**
- **Citeseer**
  
## Contributing
We welcome contributions to ABG-NAS! If you would like to contribute, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License
ABG-NAS is released under the MIT License. See the `LICENSE` file for more details.
