# PyTorch base class for handling datasets
from torch.utils.data import Dataset
# The RadioML comes in HDF5 format
import h5py


# Derive a custom dataset from the PyTorch base
class RadioMLDataset(Dataset):
    # Loads and filters the dataset index
    def __init__(self, path, classes=None, signal_to_noise_ratios=None):
        # Open the dataset file in reading mode
        self.file = h5py.File(path, "r")
        # The dataset is composed of three subfiles: the data series, the
        # modulation classes and the signal-to-noise ration of each sample
        data, cls, snr = self.file["X"], self.file["Y"], self.file["Z"]
        # All subfiles must have the same length
        assert len(data) == len(classes) == len(snr), "Dataset is corrupted"

        # Convert one-hot encoded class labels to numbers
        cls = cls[:].argmax(axis=-1)
        # First filter by desired modulation classes
        indices = {
            idx for idx, cls in enumerate(cls) if cls in classes
        }

