# KEM-Attack: Knowledge-Enhanced Metaheuristic Framework

## Description
KEM-Attack is a novel framework for hard-label textual adversarial attacks that efficiently generates adversarial examples through the synergistic integration of linguistic knowledge and metaheuristic optimization. The framework operates in two stages: (1) knowledge-enhanced substitution space construction using WordNet and HowNet, and (2) adversarial example generation using a specialized Discrete Grey Wolf Optimization algorithm.

## Dataset Information
The framework was evaluated on four widely-used text classification datasets:

| Dataset       | #Classes | Train | Test | Avg. Length | URL                                                          |
| ------------- | -------- | ----- | ---- | ----------- | ------------------------------------------------------------ |
| **MR**        | 2        | 9K    | 1K   | 20          | [Download](https://www.cs.cornell.edu/people/pabo/movie-review-data/) |
| **IMDB**      | 2        | 25K   | 25K  | 215         | [Download](https://ai.stanford.edu/~amaas/data/sentiment/)   |
| **Yelp**      | 2        | 560K  | 18K  | 152         | [Download](https://business.yelp.com/data/resources/open-dataset/) |
| **AG's News** | 4        | 120K  | 7.6K | 43          | [Download](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) |

For experiments, we randomly select 1,000 correctly classified instances from the test set of each dataset.

## Requirements
- Python >= 3.8
- PyTorch >= 0.4
- TensorFlow >= 1.0
- TensorflowHub
- NumPy
- NLTK
- OpenHowNet

## Implementation Instruction
+ Fork the repository https://github.com/RishabhMaheshwary/hard-label-attack and follow its instruction to install the environment
+ First, run the code from hard-label-attack. Then simply run the run.sh script
+ Note: The paths for some dependency files are hardcoded in the code and need to be manually changed.

## Acknowledgement
We thank the authors of https://github.com/RishabhMaheshwary/hard-label-attack for sharing their code.
