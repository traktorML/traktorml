import argparse
import random

import mlflow

FAKE_EXPERIMENTS = ["Hyperparameter tuning never works", "Big Pickle"]


def generate_run(experiment_name: str) -> None:
    """Generate a fake run for the given experiment.

    Should be expanded to include more stuff.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_param("fit_intercept", random.choice([1, 0]))
        mlflow.log_metric("MAE", random.random())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, required=False, default=100)
    args = parser.parse_args()
    for _ in range(args.n_runs):
        generate_run(random.choice(FAKE_EXPERIMENTS))
