import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from main_NLP import TransformerClassifier, load_data, train, validate
from transformers_project.linformer.linformer import LinformerEnc


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    writer,
    device,
    num_epochs=5,
    model_type="transformer",
    seq_len=64,
):
    """Train and evaluate a model for specified number of epochs while tracking metrics."""
    epoch_metrics = []

    for epoch in range(1, num_epochs + 1):
        # Training phase with timing
        model.train()
        epoch_start_time = time.time()
        train_loss, train_acc = 0, 0

        for i, batch in enumerate(
            tqdm(
                train_loader,
                desc=f"{model_type}-{seq_len} Training Epoch {epoch}",
            )
        ):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)

            # Handle output shape from LinformerEnc properly
            if model_type == "linformer":
                if logits.dim() == 2 and logits.size(1) == 2:
                    pass
                elif logits.dim() == 2 and logits.size(1) == 1:
                    logits = torch.cat([1 - logits, logits], dim=1)
                else:
                    print(f"Debug - logits shape: {logits.shape}")

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).float().mean().item()

            train_loss += loss.item()
            train_acc += acc
            writer.add_scalar(
                f"Loss/train_{model_type}_{seq_len}",
                loss,
                epoch * len(train_loader) + i,
            )
            writer.add_scalar(
                f"Accuracy/train_{model_type}_{seq_len}",
                acc,
                epoch * len(train_loader) + i,
            )

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        train_time = time.time() - epoch_start_time

        # Validation phase with timing
        model.eval()
        val_start_time = time.time()
        val_loss, val_acc = 0, 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc=f"{model_type}-{seq_len} Validation Epoch {epoch}",
            ):
                input_ids = batch["input_ids"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids)

                # Handle output shape from LinformerEnc properly
                if model_type == "linformer":
                    if logits.dim() == 2 and logits.size(1) == 2:
                        pass
                    elif logits.dim() == 2 and logits.size(1) == 1:
                        logits = torch.cat([1 - logits, logits], dim=1)
                    else:
                        print(f"Debug - logits shape: {logits.shape}")

                loss = loss_fn(logits, labels)
                preds = torch.argmax(logits, dim=-1)
                acc = (preds == labels).float().mean().item()

                val_loss += loss.item()
                val_acc += acc

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        val_time = time.time() - val_start_time

        # Record metrics for this epoch
        epoch_data = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,
            "train_time": train_time,
            "val_loss": avg_val_loss,
            "val_acc": avg_val_acc,
            "val_time": val_time,
            "total_time": train_time + val_time,
            "memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        }
        epoch_metrics.append(epoch_data)

        # Log to tensorboard
        writer.add_scalar(
            f"Loss/train_epoch_{model_type}_{seq_len}", avg_train_loss, epoch
        )
        writer.add_scalar(
            f"Accuracy/train_epoch_{model_type}_{seq_len}",
            avg_train_acc,
            epoch,
        )
        writer.add_scalar(
            f"Loss/val_epoch_{model_type}_{seq_len}", avg_val_loss, epoch
        )
        writer.add_scalar(
            f"Accuracy/val_epoch_{model_type}_{seq_len}", avg_val_acc, epoch
        )

        print(
            f"[{model_type}-{seq_len} Epoch {epoch}] "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Train Time: {train_time:.2f}s | "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, Val Time: {val_time:.2f}s"
        )

    return epoch_metrics


def compare_seq_lengths(num_epochs=5):
    """Compare training speed and performance with different sequence lengths for both models."""

    # Create results directory if it doesn't exist
    os.makedirs("benchmark_results", exist_ok=True)

    # Sequence lengths to test
    seq_lengths = [64, 128, 256, 512]
    batch_size = 64

    # Initialize results storage
    results = {
        "transformer": {
            length: {"summary": {}, "epochs": []} for length in seq_lengths
        },
        "linformer": {
            length: {"summary": {}, "epochs": []} for length in seq_lengths
        },
    }

    for seq_len in seq_lengths:
        print(f"\n=== Testing sequence length: {seq_len} ===")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dataset = load_data(tokenizer, max_length=seq_len)

        train_loader = DataLoader(
            dataset["train"], batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(dataset["validation"], batch_size=batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Test Transformer
        print(f"Testing standard Transformer with seq_len={seq_len}")
        model = TransformerClassifier(
            vocab_size=tokenizer.vocab_size, num_classes=2
        ).to(device)

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(device)

        # Run training for specified epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs * len(train_loader)
        )
        loss_fn = nn.CrossEntropyLoss()
        writer = SummaryWriter(f"runs/transformer_seq{seq_len}")

        transformer_metrics = train_and_evaluate(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_fn,
            writer,
            device,
            num_epochs=num_epochs,
            model_type="transformer",
            seq_len=seq_len,
        )

        results["transformer"][seq_len]["epochs"] = transformer_metrics
        results["transformer"][seq_len]["summary"] = {
            "final_val_acc": transformer_metrics[-1]["val_acc"],
            "total_time": sum(
                epoch["total_time"] for epoch in transformer_metrics
            ),
            "max_memory": torch.cuda.max_memory_allocated(device)
            / (1024**2),  # MB
        }

        # Test Linformer
        print(f"Testing Linformer with seq_len={seq_len}")
        model = LinformerEnc(
            seq_len=seq_len,
            dim=128,
            dim_lin_base=128,
            vocab_size=tokenizer.vocab_size,
            n_features=1,
            device=device,
            d_conversion=32,  # Projected dimension k
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

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(device)

        # Run training for specified epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs * len(train_loader)
        )
        loss_fn = nn.CrossEntropyLoss()
        writer = SummaryWriter(f"runs/linformer_seq{seq_len}")

        linformer_metrics = train_and_evaluate(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_fn,
            writer,
            device,
            num_epochs=num_epochs,
            model_type="linformer",
            seq_len=seq_len,
        )

        results["linformer"][seq_len]["epochs"] = linformer_metrics
        results["linformer"][seq_len]["summary"] = {
            "final_val_acc": linformer_metrics[-1]["val_acc"],
            "total_time": sum(
                epoch["total_time"] for epoch in linformer_metrics
            ),
            "max_memory": torch.cuda.max_memory_allocated(device)
            / (1024**2),  # MB
        }

    # Print results summary
    print("\n=== Results Summary ===")
    print(
        "Sequence Length | Model      | Val Acc | Total Time (s) | Memory (MB)"
    )
    print(
        "----------------|------------|---------|----------------|------------"
    )
    for seq_len in seq_lengths:
        t_summary = results["transformer"][seq_len]["summary"]
        l_summary = results["linformer"][seq_len]["summary"]

        print(
            f"{seq_len:14} | Transformer | {t_summary['final_val_acc']:.4f} | {t_summary['total_time']:.2f} | {t_summary['max_memory']:.2f}"
        )
        print(
            f"{seq_len:14} | Linformer   | {l_summary['final_val_acc']:.4f} | {l_summary['total_time']:.2f} | {l_summary['max_memory']:.2f}"
        )

        # Calculate speedup ratio and accuracy difference
        speedup = t_summary["total_time"] / l_summary["total_time"]
        memory_ratio = t_summary["max_memory"] / l_summary["max_memory"]
        acc_diff = l_summary["final_val_acc"] - t_summary["final_val_acc"]

        print(
            f"{seq_len:14} | Comparison  | Δ={acc_diff:.4f} | Speedup={speedup:.2f}x | Mem ratio={memory_ratio:.2f}x"
        )
        print(
            "----------------|------------|---------|----------------|------------"
        )

    # Save detailed results to file
    with open("benchmark_results/sequence_length_comparison.json", "w") as f:
        json.dump(results, f)

    # Save epoch-by-epoch data in separate files for easier plotting
    for model_type in ["transformer", "linformer"]:
        for seq_len in seq_lengths:
            epochs_data = results[model_type][seq_len]["epochs"]
            with open(
                f"benchmark_results/{model_type}_seq{seq_len}_epochs.json", "w"
            ) as f:
                json.dump(epochs_data, f)

    return results


def main():
    compare_seq_lengths(num_epochs=5)


if __name__ == "__main__":
    main()
