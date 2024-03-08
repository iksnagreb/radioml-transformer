# YAML for saving experiment metrics
import yaml
# Use the DVC api for loading the YAML parameters
import dvc.api
# Progressbar in for loops
import tqdm
# Pandas to handle the results as table, i.e., DataFrame
import pandas as pd
# PyTorch base package: Math and Tensor Stuff
import torch
# Loads shuffled batches from datasets
from torch.utils.data import DataLoader
# Classification metrics
from sklearn.metrics import accuracy_score

# The RadioML modulation classification transformer model
from model import RadioMLTransformer
# The RadioML modulation classification dataset
from dataset import get_datasets
# Seeding RNGs for reproducibility
from utils import seed


# Main evaluation loop: Takes a trained model, loads the dataset and sets and
# runs a bunch of inferences to collect metrics
def evaluate(model, dataset, batch_size, loader):  # noqa: Shadows model
    # Check whether GPU eval is available and select the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Move the model to the training device
    model = model.to(device)  # noqa: Shadows model...
    # Set model to evaluation mode
    model = model.eval()  # noqa: Shadows model...

    # Load the RadioML dataset splits as configured
    _, _, eval_data = get_datasets(**dataset)
    # Create a batched and shuffled data loader for each of the dataset splits
    eval_data = DataLoader(eval_data, batch_size=batch_size, **loader)

    # Start collecting predictions and ground-truth in a data frame
    results = []

    # Evaluation requires no gradients
    with torch.no_grad():
        # Iterate the (input, target labels) pairs
        for x, cls, snr in tqdm.tqdm(eval_data, "eval-batch", leave=False):
            # Model forward pass producing logits corresponding to class
            # probabilities. Aggregate the logits (~ class probabilities) along
            # the sequence dimension.
            p = model(x.to(device)).sum(dim=1).argmax(dim=-1)
            # Each sample in the batch will generate a new record
            for cls, p, snr in zip(cls, p, snr):  # noqa: Shadows cls, p, snr
                # Append prediction and ground truth to the data frame
                results.append({
                    "cls": cls.item(), "prediction": p.item(), "snr": snr.item()
                })

    # Convert list of records to pandas data frame to do statistics to it
    results = pd.DataFrame.from_records(results)

    # Computes the prediction accuracy over the class and predicted class of a
    # data frame or a group of a data frame
    def accuracy(group):
        return accuracy_score(*group[["cls", "prediction"]].T.values)

    # Accuracy per Signal-to-Noise ratio
    accuracy_per_snr = results.groupby("snr").apply(  # noqa: Shadows
        accuracy, include_groups=False
    )
    # Properly format this as a dictionary which can be understood for plotting
    # by dvc
    accuracy_per_snr = [  # noqa: Shadows from outer scope
        {"snr": snr, "acc": acc} for snr, acc in accuracy_per_snr.items()
    ]

    # Collect true and predicted labels for creating a confusion matrix plot
    classes = results[["cls", "prediction"]]  # noqa: Shadows from outer scope

    # Compute the classification accuracy over the evaluation dataset
    return {"accuracy": accuracy(results)}, accuracy_per_snr, classes


# Script entrypoint
if __name__ == "__main__":
    # Load the parameters file
    params = dvc.api.params_show("params.yaml", stages="eval")
    # Seed all RNGs
    seed(params["seed"])
    # Create a new model instance according to the configuration
    model = RadioMLTransformer(**params["model"])
    # Load the trained model parameters
    model.load_state_dict(torch.load("outputs/model.pt"))
    # Pass the model and the evaluation configuration to the evaluation loop
    metrics, accuracy_per_snr, classes = evaluate(
        model, dataset=params["dataset"], **params["eval"]
    )
    # Dump the metrics dictionary as yaml
    with open("metrics.yaml", "w") as file:
        # Dictionary which can be dumped into YAML
        yaml.safe_dump(metrics, file)
    # Save the accuracy grouped per SNR into a separate yaml to serve this as a
    # plot
    with open("accuracy-per-snr.yaml", "w") as file:
        # Dictionary which can be dumped into YAML
        yaml.safe_dump({"Accuracy per SNR": accuracy_per_snr}, file)
    # Save the confusion matrix into a separate yaml to serve this as a
    # plot
    #   Note: Save this as CSV, as YAML stores excessive metadata
    with open("classes.csv", "w") as file:
        # Dictionary which can be dumped into YAML
        file.write(classes.to_csv(index=False))
