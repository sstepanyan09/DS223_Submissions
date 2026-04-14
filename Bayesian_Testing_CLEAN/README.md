# Homework 2: A/B Testing with Bandit Algorithms

This project implements two multi-armed bandit algorithms for an A/B testing scenario:

* Epsilon-Greedy
* Thompson Sampling

The experiment uses four advertisement options with true rewards:

`BANDIT_REWARD = [1, 2, 3, 4]`

and runs for:

`NUMBER_OF_TRIALS = 20000`

## Project Structure

* `Bandit.py` — main implementation of the abstract `Bandit` class, `EpsilonGreedy`, `ThompsonSampling`, visualization methods, and comparison logic
* `HW2.ipynb` — notebook deliverable that imports `Bandit.py` and runs the experiment
* `requirements.txt` — required package versions
* `README.md` — project description and usage instructions

## Implemented Tasks

This project includes:

1. A `Bandit` abstract base class
2. An `EpsilonGreedy` class with decaying epsilon `1/t`
3. A `ThompsonSampling` class with known precision
4. Visualization of:

   * the learning process of both algorithms
   * cumulative rewards
   * cumulative regrets
5. CSV export of results in the format:

   * `Bandit`
   * `Reward`
   * `Algorithm`

## Output Files

Running the project generates the following files:

* `epsilon_greedy_rewards.csv`
* `thompson_sampling_rewards.csv`
* `combined_rewards.csv`

## How to Run

### Option 1: Run the notebook

Open `HW2.ipynb` and execute the cells.

### Option 2: Run the Python file

```bash
python Bandit.py
```

## Requirements

Install the required package with:

```bash
pip install -r requirements.txt
```

## Libraries Used

* `loguru`
* `numpy`
* `pandas`
* `matplotlib`

## Notes

* Logging is used instead of plain `print()` statements where appropriate.
* The notebook is the final deliverable for submission.
* The implementation follows the structure provided by the instructor in `Bandit.py`.

