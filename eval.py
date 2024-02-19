# YAML for loading experiment configurations
import yaml
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
            # probabilities
            p = model(x.to(device)).argmax(dim=-1)
            # Each sample in the batch will generate a new record
            for cls, p, snr in zip(cls, p, snr):  # noqa: Shadows cls, p, snr
                # Append prediction and ground truth to the data frame
                results.append({
                    "cls": cls.item(), "prediction": p.item(), "snr": snr.item()
                })

    # Convert list of records to pandas data frame to do statistics to it
    results = pd.DataFrame.from_records(results)
    # Compute the classification accuracy over the evaluation dataset
    return {
        "accuracy": accuracy_score(*results[["cls", "prediction"]].T.values)
    }


# Script entrypoint
if __name__ == "__main__":
    # Open the configuration file
    with open("params.yaml") as file:
        # Load the configuration from yaml format
        params = yaml.safe_load(file)
    # Seed all RNGs
    seed(params["seed"])
    # Create a new model instance according to the configuration
    model = RadioMLTransformer(**params["model"])
    # Load the trained model parameters
    model.load_state_dict(torch.load("outputs/model.pt"))
    # Pass the model and the evaluation configuration to the evaluation loop
    metrics = evaluate(model, dataset=params["dataset"], **params["eval"])
    # Dump the metrics dictionary as yaml
    with open("metrics.yaml", "w") as file:
        # Dictionary which can be dumped into YAML
        yaml.safe_dump(metrics, file)
