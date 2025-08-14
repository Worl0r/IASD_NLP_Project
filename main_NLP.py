import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformers_project.linformer.linformer import LinformerEnc


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes=2,
        d_model=128,
        nhead=2,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.2,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = (
            torch.arange(0, seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand_as(input_ids)
        )
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # No need to transpose since batch_first=True
        x = self.transformer(x)  # (batch, seq_len, d_model)

        cls_token = x[:, 0, :]  # Use [CLS] token
        logits = self.classifier(cls_token)
        return logits


# Load SST-2 dataset
def load_data(tokenizer, max_length=64, train_sample_ratio=0.5):
    dataset = load_dataset("glue", "sst2")

    def tokenize(example):
        return tokenizer(
            example["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(tokenize, batched=True)

    # Calculate the number of examples to keep
    num_train_examples = int(len(dataset["train"]) * train_sample_ratio)
    # Randomly select indices to keep
    indices = torch.randperm(len(dataset["train"]))[
        :num_train_examples
    ].tolist()
    # Create a subset of the training data
    dataset["train"] = dataset["train"].select(indices)
    print(
        f"Reduced training dataset to {num_train_examples} examples ({train_sample_ratio * 100:.1f}% of original)"
    )

    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    return dataset


def train(
    model, dataloader, optimizer, scheduler, loss_fn, writer, device, epoch
):
    model.train()
    total_loss, total_acc = 0, 0

    for i, batch in enumerate(
        tqdm(dataloader, desc=f"Training Epoch {epoch}")
    ):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)

        # Handle output shape from LinformerEnc properly
        if isinstance(model, LinformerEnc):
            # Ensure logits have shape [batch_size, num_classes]
            if logits.dim() == 2 and logits.size(1) == 2:
                # Already correct shape
                pass
            elif logits.dim() == 2 and logits.size(1) == 1:
                # Need to convert to 2-class output
                logits = torch.cat([1 - logits, logits], dim=1)
            else:
                # Unexpected shape
                raise ValueError(f"Unexpected logits shape: {logits.shape}")

            loss = loss_fn(logits, labels)
        else:
            # Original handling for TransformerClassifier
            loss = loss_fn(
                logits, labels
            )  # Remove .squeeze(-1) and .to(float)

        loss.backward()
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean().item()

        total_loss += loss.item()
        total_acc += acc
        writer.add_scalar("Loss/train_t", loss, epoch + i * len(dataloader))
        writer.add_scalar("Accuracy/train_t", acc, epoch + i * len(dataloader))

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/train", avg_acc, epoch)


def validate(model, dataloader, loss_fn, writer, device, epoch):
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation Epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)

            # Handle output shape from LinformerEnc properly
            if isinstance(model, LinformerEnc):
                # Ensure logits have shape [batch_size, num_classes]
                if logits.dim() == 2 and logits.size(1) == 2:
                    # Already correct shape
                    pass
                elif logits.dim() == 2 and logits.size(1) == 1:
                    # Need to convert to 2-class output
                    logits = torch.cat([1 - logits, logits], dim=1)
                else:
                    # Unexpected shape
                    print(f"Debug - logits shape: {logits.shape}")

                loss = loss_fn(logits, labels)
            else:
                # Original handling for TransformerClassifier
                loss = loss_fn(
                    logits, labels
                )  # Remove .squeeze(-1) and .to(float)

            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).float().mean().item()

            total_loss += loss.item()
            total_acc += acc

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Accuracy/val", avg_acc, epoch)
    print(f"[Epoch {epoch}] Val Loss: {avg_loss:.4f}, Val Acc: {avg_acc:.4f}")


def main():
    seq_len = 64  # tronque si phrase plus longue. SST2 mediane c'est env 100 token, le max c'est 244
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_data(tokenizer, max_length=seq_len)

    # Configuration
    num_epochs = 10

    batch_size = 64
    train_loader = DataLoader(
        dataset["train"], batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"------{device}------")
    print(f"seq_len: {seq_len} ; batch_size: {batch_size}")

    # model = TransformerClassifier(
    #     vocab_size=tokenizer.vocab_size, num_classes=2
    # ).to(device)

    model = LinformerEnc(
        seq_len=seq_len,
        dim=128,
        dim_lin_base=128,
        vocab_size=tokenizer.vocab_size,
        n_features=1,
        device=device,
        d_conversion=32,
        max_prediction_length=1,
        dropout_input=0.1,
        dropout_multi_head_att=0.1,
        dropout_lin_att=0.1,
        dim_ff=128,
        ff_intermediate=None,
        dropout_ff=0.1,
        nhead=2,
        n_layers=4,
        dropout=0.2,
        parameter_sharing="layerwise",
        k_reduce_by_layer=0,
        method="learnable",
        activation="gelu",
    ).to(device)

    # Print number of parameters : linformer=4320513 ; transformer=4502530
    # import numpy as np
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter("runs/transformer_classifier")

    for epoch in range(1, num_epochs + 1):
        train(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            writer,
            device,
            epoch,
        )
        validate(model, val_loader, loss_fn, writer, device, epoch)

    writer.close()


if __name__ == "__main__":
    main()
