from tqdm import tqdm

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader


def predict(model: nn.Module, dataloader: DataLoader, device) -> dict[str, any]:
    model.eval()
    model.to(device)

    predictions = []
    # Disable gradient computation.
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predict"):
            # Move abtch to device.
            batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
            output = model(**batch)
            # Remove loss.
            output.pop("loss")
            output = {k: v.tolist() if isinstance(v, Tensor) else v for k, v in output.items()}
            #print(output)
            # Convert dict of lists to list of dicts.
            output_unfolded = [dict(zip(output.keys(), t)) for t in zip(*output.values())]
            predictions.extend(output_unfolded)
    return predictions

