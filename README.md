# autoresearch-classical-ml

Code & notes on the Autoresearch for Classical ML project

## Project description

**Topic**: Autoresearch for Classical ML project 

**Description**: We wish to estimate how well agentic systems are able to improve performance of classical ML pipelines.

## Research objective

### Abstract

It is well known that agentic systems are able to solve end-to-end coding problems via stable feature set expansion and bug fixing. This scenario confirms their ability in the feature expansion direction. There is however another important scenario - performance improvement for the existing set of features. The performance direction is huge in volume. In this project, we narrow it down to the problems typical for classical machine learning.

### Key research questions
- **RQ1:** how many optimization iterations can the agentic system support performance improvement until reaching a plateau?
- **RQ2:** what types of interventions (hyperparameter tuning, feature engineering, model selection, pipeline restructuring) are most effective for agentic improvement?
- **RQ3:** what theoretical and practical ML analysis tools can the agentic system come up with without external guidance?

### Why this deserves studying

- **Feature expansion ≠ performance optimization**: Agentic systems are well-documented for expanding capabilities (new features, bug fixes), but performance *on existing features* is a distinct and larger problem space in practice
- **Practical demand**: Classical ML pipelines dominate production systems; automating their optimization would have immediate real-world impact
- **AutoML gap**: Understanding whether agents can autonomously generate theoretical/practical ML insights pushes beyond pipeline tweaking into genuine scientific automation
- **Human-AI collaboration design**: Knowing when agents plateau and which interventions work is critical for designing effective hybrid workflows where humans add value at the right moments
- **Efficiency**: Understanding optimization trajectories helps build more sample-efficient agentic systems, reducing compute and cost
