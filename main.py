import os
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.utils import get_logger, read_yaml_config

from datasets import load_dataset
from dataset import preprocessing_fn, DataCollator
from transformers import BertTokenizer
from torch.utils.data import DataLoader

# Define the logger
logger = get_logger()

# Define the device
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Define the tensor dashboard
writer = SummaryWriter("runs/")

# Configuration
config_path = os.path.join(os.getcwd(), "configuration.yaml")


def main_NLP():
    # Read the configuration
    config = read_yaml_config(config_path)

    n_samples = config["DATA"]["n_samples"]
    test_ratio = config["DATA"]["ratio"]
    batch_size = config["DATA"]["batch_size"]
    epochs = config["TRAINING"]["epochs"]

    # Load the dataset
    dataset = load_dataset("stanfordnlp/imdb", split="train")

    # Tokenize
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True
    )

    # We shuffle the data
    shuffled_dataset = dataset.shuffle(seed=42)

    # Select n_sampless samples
    small_dataset = shuffled_dataset.select(list(range(n_samples)))

    dataset = small_dataset.train_test_split(test_size=test_ratio)

    def preprocess_text(x):
        ids = tokenizer(
            x["description"], truncation=True, max_length=256, padding=False
        )["input_ids"]
        return {"input_ids": ids, "label": x["label"] - 1}

    # Clean the dataset and tokenize it directly
    dataset = dataset.map(preprocess_text)

    data_collator = DataCollator(tokenizer)

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )

    print(dataset)


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

if __name__() == "main":
    main_NLP()


def validation_step(valid_dataloader, model, criterion):
    n_valid = len(valid_dataloader.dataset)
    model.eval()
    total_loss = 0.0
    correct = 0
    n_iter = 0
    with torch.no_grad():
        for batch in valid_dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            pad_mask = batch["pad_mask"].to(DEVICE)
            output = model(input_ids, pad_mask)
            loss = criterion(output, labels)
            total_loss += loss.item()
            correct += (output.argmax(axis=-1) == labels).sum().item()
            n_iter += 1
    return total_loss / n_iter, correct / n_valid


def train_one_epoch(train_dataloader, model, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    n_train = len(train_dataloader.dataset)
    n_iter = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        pad_mask = batch["pad_mask"].to(DEVICE)
        class_scores = model(input_ids, pad_mask)  # (B, 4)

        loss = criterion(class_scores, labels)  # scalaire (1,)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (class_scores.argmax(axis=-1) == labels).sum().item()
        n_iter += 1

    return total_loss / n_iter, correct / n_train


def train(model, train_dataloader, valid_dataloader, lr=0.01, n_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Track training loss, training accuracy, validation loss and validation accuracy and plot in the end
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    model.to(DEVICE)
    for epoch in tqdm(range(n_epochs)):
        train_loss, train_accuracy = train_one_epoch(
            train_dataloader, model, optimizer, criterion
        )
        valid_loss, valid_accuracy = validation_step(
            valid_dataloader, model, criterion
        )
        print(
            f"Epoch {epoch + 1}: train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}, valid_loss: {valid_loss:.4f}, valid_accuracy: {valid_accuracy:.4f}"
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="train loss")
    plt.plot(valid_losses, label="valid loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="train accuracy")
    plt.plot(valid_accuracies, label="valid accuracy")
    plt.legend()
