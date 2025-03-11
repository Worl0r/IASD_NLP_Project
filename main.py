import os
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.utils import get_logger, read_yaml_config

from datasets import load_dataset
from transformers import BertTokenizer

# Define the logger
logger = get_logger()

# Define the device
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Define the tensor dashboard
writer = SummaryWriter("runs/")

# Configuration
config_path = os.path.join(os.getcwd(), "configuration.yaml")


def main():
    # Read the configuration
    config = read_yaml_config(config_path)

    n_samples = config["DATA"]["n_samples"]

    # Load the dataset
    dataset = load_dataset("stanfordnlp/imdb", split="train")

    # Tokenize
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True
    )

    # We shuffle the data
    shuffled_dataset = dataset.shuffle(seed=42)

    # Select n_sampless samples
    small_dataset = dataset.select(list(range(n_samples)))
    print(dataset)


epochs = 5


def train(
    model,
    loader,
    epoch,
    optimizer,
    criterion,
    writer,
    log_interval=10,
):
    model.train()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        writer.add_scalar(
            "Training", loss.cpu().item(), epoch * len(loader) + batch_idx
        )

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * images.shape[0],
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )


@torch.no_grad()
def validate(
    model,
    loader,
    loss_vector,
    criterion,
    accuracy_vector,
    epoch,
    step="Validation",
    writer=None,
):
    model.eval()

    val_loss, correct = 0, 0
    nbr_examples = len(next(iter(loader))) * len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        val_loss = criterion(output, labels).item()

        pred_softmax = nn.Softmax()(output)

        # Passez les outputs logits
        pred = torch.max(pred_softmax, dim=-1).indices

        # Calculez le nombre de prédiction correctes et on ajoute à la liste correct
        correct += torch.eq(pred, labels).sum()

        loss_vector.append(val_loss)

        # Calculez l'accuracy, formule  100 * "predictions correctes" / "nombre d'exemples"
        accuracy = 100 * correct / nbr_examples
        accuracy_vector.append(accuracy)
        writer.add_scalar(step, val_loss, epoch * images.shape[0] + batch_idx)
        writer.add_scalar(
            step, accuracy.cpu().item(), epoch * images.shape[0] + batch_idx
        )


for epoch in range(1, epochs + 1):
    train(
        resnet,
        train_loader,
        epoch,
        optimizer,
        criterion,
        writer,
        log_interval=1,
    )

    validate(
        resnet,
        train_loader,
        loss_validation,
        criterion,
        acc_validation,
        epoch,
        step="Training",
        writer=writer,
    )
    validate(
        resnet,
        test_loader,
        loss_validation,
        criterion,
        acc_validation,
        epoch,
        step="Validation",
        writer=writer,
    )

writer.close()

if __init__() == "main":
    main()
