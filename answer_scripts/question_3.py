import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torchvision import datasets, models
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

imagenet_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
data_transforms = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(), imagenet_norm]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(), imagenet_norm]),
}

data_dir = os.path.join("data", "hymenoptera_data")
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ["train", "val"]}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=(x == "train"), num_workers=0)
               for x in ["train", "val"]}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes
print(dataset_sizes, class_names)


MODEL_NAME = "resnet18"
MODEL_DIR = os.path.join(BASE_DIR, MODEL_NAME)          # folder named after the model
WEIGHTS_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_imagenet1k_v1.pth")


def load_pretrained_resnet18():
    """Load ImageNet1K ResNet18 from the local folder; download and save it there if missing."""
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading pretrained weights from {WEIGHTS_PATH}")
        model = models.resnet18(weights=None)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    else:
        print("Pretrained weights not found locally, downloading...")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f"Saved weights to {WEIGHTS_PATH}")
    return model


def fmt_time(seconds):
    m, s = divmod(round(seconds), 60)
    return f"{m}m {s}s"


def build_finetune():
    model = load_pretrained_resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(class_names))   # 2 classes
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)   # all layers trained
    return model, optimizer


def build_feature_extractor():
    model = load_pretrained_resnet18()
    for p in model.parameters():
        p.requires_grad = False                                    # freeze the backbone
    model.fc = nn.Linear(model.fc.in_features, len(class_names))   # new layer is trainable
    model = model.to(device)
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)   # only fc trained
    return model, optimizer


def train_resnet(model, optimizer, scheduler, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_wts = 0.0, copy.deepcopy(model.state_dict())
    since = time.time()

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (outputs.argmax(dim=1) == labels).sum().item()

            if phase == "train":
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = 100.0 * running_corrects / dataset_sizes[phase]
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            if phase == "val" and epoch_acc > best_acc:
                best_acc, best_wts = epoch_acc, copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"train loss {history['train_loss'][-1]:.4f}, acc {history['train_acc'][-1]:.2f}% | "
              f"val loss {history['val_loss'][-1]:.4f}, acc {history['val_acc'][-1]:.2f}%")

    elapsed = time.time() - since
    model.load_state_dict(best_wts)          # keep the best-validation weights
    return model, history, best_acc, elapsed


def run_transfer(build_fn, name, num_epochs=10, seed=0):
    torch.manual_seed(seed)
    model, optimizer = build_fn()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- {name} | trainable parameters: {trainable:,} ---")
    _, history, best_acc, elapsed = train_resnet(model, optimizer, scheduler, num_epochs)
    print(f"Training time: {fmt_time(elapsed)} | best val acc: {best_acc:.2f}%")
    return {"history": history, "best_val_acc": best_acc, "time": elapsed, "trainable": trainable}


def plot_transfer(history, name):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, history["train_loss"], marker="o", label="Train")
    ax[0].plot(ep, history["val_loss"], marker="s", label="Validation")
    ax[0].set_title(f"{name}: loss per epoch")
    ax[0].set_ylabel("Loss")
    ax[1].plot(ep, history["train_acc"], marker="o", label="Train")
    ax[1].plot(ep, history["val_acc"], marker="s", label="Validation")
    ax[1].set_title(f"{name}: accuracy per epoch")
    ax[1].set_ylabel("Accuracy (%)")
    for a in ax:
        a.set_xlabel("Epoch")
        a.grid(True)
        a.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    res_ft = run_transfer(build_finetune, "Fine-tuning")
    plot_transfer(res_ft["history"], "Fine-tuning")

    res_fe = run_transfer(build_feature_extractor, "Feature extractor")
    plot_transfer(res_fe["history"], "Feature extractor")

    print("\n=== Summary ===")
    for name, r in [("Fine-tuning", res_ft), ("Feature extractor", res_fe)]:
        h = r["history"]
        print(f"{name}: trainable={r['trainable']:,}, time={fmt_time(r['time'])}, "
              f"final train acc={h['train_acc'][-1]:.2f}%, final val acc={h['val_acc'][-1]:.2f}%, "
              f"best val acc={r['best_val_acc']:.2f}%")