import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from ema_pytorch import EMA
from pandas.core.window.doc import kwargs_scipy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F

import wandb
from models import get_model
from scheduler import CosineAnnealingWithWarmRestartsLR

seed = 2001
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(
        self,
        model_a,
        model_b,
        training_dataloader,
        validation_dataloader,
        testing_dataloader,
        classes,
        output_dir,
        max_epochs: int = 10000,
        early_stopping_patience: int = 12,
        execution_name=None,
        lr: float = 1e-3,
        amp: bool = False,
        ema_decay: float = 0.99,
        ema_update_every: int = 16,
        gradient_accumulation_steps: int = 1,
        checkpoint_path: str = None,
        temperature: float = 4.0,
        lambda_kd: float = 0.5
    ):
        self.epochs = max_epochs

        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.testing_dataloader = testing_dataloader

        self.classes = classes
        self.num_classes = len(classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device used: " + self.device.type)

        self.amp = amp
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.model_a = model_a.to(self.device)
        self.model_b = model_b.to(self.device)
        self.optimizer_a = AdamW(self.model_a.parameters(), lr=lr)
        self.optimizer_b = AdamW(self.model_b.parameters(), lr=1e-4)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.scheduler_a = CosineAnnealingWithWarmRestartsLR(
            self.optimizer_a, warmup_steps=128, cycle_steps=1024
        )
        self.scheduler_b = CosineAnnealingWithWarmRestartsLR(
            self.optimizer_b, warmup_steps=128, cycle_steps=1024
        )
        self.ema_a = EMA(self.model_a, beta=ema_decay, update_every=ema_update_every).to(self.device)
        self.ema_b = EMA(self.model_b, beta=ema_decay, update_every=ema_update_every).to(self.device)

        self.early_stopping_patience = early_stopping_patience

        self.output_directory = Path(output_dir)
        self.output_directory.mkdir(exist_ok=True)

        self.best_val_accuracy = 0

        self.execution_name = "model" if execution_name is None else execution_name

        self.temperature = temperature
        self.lambda_kd = lambda_kd

        if checkpoint_path:
            self.load(checkpoint_path)

        wandb.watch(model_a, log="all")
        wandb.watch(model_b, log="all")

    def run(self):
        counter = 0  # Counter for epochs with no validation loss improvement

        # log a batch of images
        images, _ = next(iter(self.training_dataloader))
        images = [transforms.ToPILImage()(image) for image in images]
        wandb.log({"Images": [wandb.Image(image) for image in images]})

        for epoch in range(self.epochs):
            print(f"[Epoch: {epoch + 1}/{self.epochs}]")

            self.visualize_stn()

            # train + val trả về kết quả 2 model
            train_loss_a, train_acc_a, train_loss_b, train_acc_b = self.train_epoch()
            val_loss_a, val_acc_a, val_loss_b, val_acc_b = self.val_epoch()

            # log cả 2 model
            wandb.log({
                "Train Loss A": train_loss_a,
                "Train Acc A": train_acc_a,
                "Train Loss B": train_loss_b,
                "Train Acc B": train_acc_b,
                "Val Loss A": val_loss_a,
                "Val Acc A": val_acc_a,
                "Val Loss B": val_loss_b,
                "Val Acc B": val_acc_b,
                "Epoch": epoch + 1,
            })

            # early stopping & save: theo student (B)
            if val_acc_b > self.best_val_accuracy:
                self.save()
                counter = 0
                self.best_val_accuracy = val_acc_b
            else:
                counter += 1
                if counter >= self.early_stopping_patience:
                    print(
                        f"Validation acc did not improve for {self.early_stopping_patience} epochs. Stopping training.")
                    break

            # test student (EMA version)
            self.test_model()

        wandb.finish()

    def train_epoch(self):
        self.model_a.train()
        self.model_b.train()

        avg_loss_a, avg_loss_b = [], []
        avg_acc_a, avg_acc_b = [], []

        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.training_dataloader))
        for batch_idx, (inputs, labels) in enumerate(self.training_dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.autocast(self.device.type, enabled=self.amp):
                logits_a, predictions_a, loss_a = self.model_a(inputs, labels)
                logits_b, predictions_b, loss_b = self.model_b(inputs, labels)

                # soft targets
                T = self.temperature
                pa = F.softmax(predictions_a / T, dim=1).detach()
                pb = F.softmax(predictions_b / T, dim=1).detach()

                # KL divergence
                kl_ab = F.kl_div(F.log_softmax(predictions_a / T, dim=1), pa, reduction="batchmean") * (T * T)
                kl_ba = F.kl_div(F.log_softmax(predictions_b / T, dim=1), pb, reduction="batchmean") * (T * T)

                # total losses
                loss_a = loss_a + self.lambda_kd * kl_ba
                loss_b = loss_b + self.lambda_kd * kl_ab
                loss = loss_a + loss_b

            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(list(self.model_a.parameters()) + list(self.model_b.parameters()), 1.0)
                self.scaler.step(self.optimizer_a)
                self.scaler.step(self.optimizer_b)
                self.optimizer_a.zero_grad(set_to_none=True)
                self.optimizer_b.zero_grad(set_to_none=True)
                self.scaler.update()
                self.ema_a.update()
                self.ema_b.update()
                self.scheduler_a.step()
                self.scheduler_b.step()

            acc_a = (logits_a == labels).float().mean().item() / labels.size(0)
            acc_b = (logits_b == labels).float().mean().item() / labels.size(0)
            avg_loss_a.append(loss_a.item())
            avg_loss_b.append(loss_b.item())
            avg_acc_a.append(acc_a)
            avg_acc_b.append(acc_b)

            pbar.set_postfix({
                "loss_a": np.mean(avg_loss_a),
                "acc_a": np.mean(avg_acc_a) * 100,
                "loss_b": np.mean(avg_loss_b),
                "acc_b": np.mean(avg_acc_b) * 100
            })
            pbar.update(1)

        pbar.close()
        return np.mean(avg_loss_a), np.mean(avg_acc_a) * 100, np.mean(avg_loss_b), np.mean(avg_acc_b) * 100

    def val_epoch(self):
        self.model_a.eval()
        self.model_b.eval()

        avg_loss_a, avg_loss_b = [], []
        preds_a, preds_b, true_labels = [], [], []

        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.validation_dataloader))
        for batch_idx, (inputs, labels) in enumerate(self.validation_dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.autocast(self.device.type, enabled=self.amp):
                logits_a, _, loss_a = self.model_a(inputs, labels)
                logits_b, _, loss_b = self.model_b(inputs, labels)

            avg_loss_a.append(loss_a.item())
            avg_loss_b.append(loss_b.item())

            preds_a.extend(logits_a.tolist())
            preds_b.extend(logits_b.tolist())
            true_labels.extend(labels.tolist())

            pbar.update(1)
        pbar.close()

        acc_a = (torch.tensor(preds_a) == torch.tensor(true_labels)).float().mean().item()
        acc_b = (torch.tensor(preds_b) == torch.tensor(true_labels)).float().mean().item()

        print(f"Val A Loss: {np.mean(avg_loss_a):.4f}, Acc: {acc_a * 100:.2f}%")
        print(f"Val B Loss: {np.mean(avg_loss_b):.4f}, Acc: {acc_b * 100:.2f}%")

        wandb.log({
            "Val Loss A": np.mean(avg_loss_a),
            "Val Acc A": acc_a * 100,
            "Val Loss B": np.mean(avg_loss_b),
            "Val Acc B": acc_b * 100,
            "confusion_matrix_b": wandb.plot.confusion_matrix(
                probs=None, y_true=true_labels, preds=preds_b, class_names=self.classes
            )
        })

        return np.mean(avg_loss_a), acc_a * 100, np.mean(avg_loss_b), acc_b * 100

    def test_model(self):
        self.ema_a.eval()

        predicted_labels = []
        true_labels = []

        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.testing_dataloader))
        for batch_idx, (inputs, labels) in enumerate(self.testing_dataloader):
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type, enabled=self.amp):
                _, logits = self.ema_a(inputs)
            outputs_avg = logits.view(bs, ncrops, -1).mean(1)
            predictions = torch.argmax(outputs_avg, dim=1)

            predicted_labels.extend(predictions.tolist())
            true_labels.extend(labels.tolist())

            pbar.update(1)

        pbar.close()

        accuracy = (
            torch.eq(torch.tensor(predicted_labels), torch.tensor(true_labels))
            .float()
            .mean()
            .item()
        )
        print("Test Accuracy: %.4f %%" % (accuracy * 100.0))

        wandb.log(
            {
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=true_labels,
                    preds=predicted_labels,
                    class_names=self.classes,
                )
            }
        )

    def visualize_stn(self):
        self.model_a.eval()  # EmoNeXt

        batch = torch.utils.data.Subset(val_dataset, range(32))
        batch = torch.stack([batch[i][0] for i in range(len(batch))]).to(self.device)

        with torch.autocast(self.device.type, enabled=self.amp):
            stn_batch = self.model_a.stn(batch)

        to_pil = transforms.ToPILImage()
        grid = to_pil(torchvision.utils.make_grid(batch, nrow=16, padding=4))
        stn_batch = to_pil(torchvision.utils.make_grid(stn_batch, nrow=16, padding=4))

        wandb.log({"batch": wandb.Image(grid), "stn": wandb.Image(stn_batch)})

    def save(self):
        data = {
            "model_a": self.model_a.state_dict(),
            "model_b": self.model_b.state_dict(),
            "opt_a": self.optimizer_a.state_dict(),
            "opt_b": self.optimizer_b.state_dict(),
            "ema_b": self.ema_b.state_dict(),  # chỉ giữ EMA student
            "scaler": self.scaler.state_dict(),
            "scheduler_a": self.scheduler_a.state_dict(),
            "scheduler_b": self.scheduler_b.state_dict(),
            "best_acc_b": self.best_val_accuracy,
        }
        torch.save(data, str(self.output_directory / f"{self.execution_name}.pt"))

    def load(self, path):
        data = torch.load(path, map_location=self.device)

        self.model_a.load_state_dict(data["model_a"])
        self.model_b.load_state_dict(data["model_b"])
        self.optimizer_a.load_state_dict(data["opt_a"])
        self.optimizer_b.load_state_dict(data["opt_b"])
        self.ema_b.load_state_dict(data["ema_b"])
        self.scaler.load_state_dict(data["scaler"])
        self.scheduler_a.load_state_dict(data["scheduler_a"])
        self.scheduler_b.load_state_dict(data["scheduler_b"])
        self.best_val_accuracy = data["best_acc_b"]


def plot_images():
    # Create a grid of images for visualization
    num_rows = 4
    num_cols = 8
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5))

    # Plot the images
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j  # Calculate the corresponding index in the dataset
            image, _ = train_dataset[index]  # Get the image
            axes[i, j].imshow(
                image.permute(1, 2, 0)
            )  # Convert tensor to PIL image format and plot
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig("images.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EmoNeXt on Fer2013")

    parser.add_argument("--dataset-path", type=str, default='FER2013', help="Path to the dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out",
        help="Path where the best model will be saved",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable mixed precision training",
    )
    parser.add_argument("--in_22k", action="store_true", default=False)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating the model weights",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="The number of subprocesses to use for data loading."
        "0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file for resuming training or performing inference",
    )
    parser.add_argument(
        "--model-size",
        choices=["tiny", "small", "base", "large", "xlarge"],
        default="tiny",
        help="Choose the size of the model: tiny, small, base, large, or xlarge",
    )

    opt = parser.parse_args()
    print(opt)

    current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    exec_name = f"EmoNeXt_{opt.model_size}_{current_time}"

    wandb.init(project="EmoNeXt", name=exec_name, anonymous="must")

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Grayscale(),
            transforms.Resize(208),
            transforms.RandomRotation(degrees=20),
            transforms.RandomCrop(196),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(208),
            transforms.RandomCrop(196),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(208),
            transforms.TenCrop(196),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops]
                )
            ),
            transforms.Lambda(
                lambda crops: torch.stack([crop.repeat(3, 1, 1) for crop in crops])
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(opt.dataset_path + "/train", train_transform)
    val_dataset = datasets.ImageFolder(opt.dataset_path + "/val", val_transform)
    test_dataset = datasets.ImageFolder(opt.dataset_path + "/test", test_transform)

    print("Using %d images for training." % len(train_dataset))
    print("Using %d images for evaluation." % len(val_dataset))
    print("Using %d images for testing." % len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model_a = get_model(len(train_dataset.classes), opt.model_size, in_22k=opt.in_22k)
    from convnext import convnext_tiny # bạn phải có hàm này
    model_b = convnext_tiny(pretrained=True, num_classes=len(train_dataset.classes))

    Trainer(
        model_a=model_a,
        model_b=model_b,
        training_dataloader=train_loader,
        validation_dataloader=val_loader,
        testing_dataloader=test_loader,
        classes=train_dataset.classes,
        execution_name=exec_name,
        lr=opt.lr,
        output_dir=opt.output_dir,
        checkpoint_path=opt.checkpoint,
        max_epochs=opt.epochs,
        amp=opt.amp,
        temperature=4.0,
        lambda_kd=0.5,
    ).run()
