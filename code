import os
import pickle, warnings, random
from pathlib import Path

import wfdb, numpy as np, neurokit2 as nk
import torch
from torch import nn
from torch.utils import data
from IPython import display
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from d2l import torch as d2l
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import csv

# ================= Global plotting style ================= #
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11

# ===================== Random Seed ===================== #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # Reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(_):
    np.random.seed(SEED)
    random.seed(SEED)

# ===================== Constants ===================== #
ROOT        = Path("/Users/datasets/mitbih")
RAW_DIR     = ROOT                      # WFDB data download root
MITDB_DIR   = RAW_DIR / "mitdb"         # Dataset directory
TRAIN_DIR   = ROOT / "train"            # Train beats (.npy)
TEST_DIR    = ROOT / "test"             # Test beats (.npy)
LABELS_PKL  = ROOT / "labels.pkl"

# Signal-related constants
LEADS       = [0, 1]      # MLII, V5
ORIG_FS     = 360         # MIT-BIH original sampling rate
TARGET_FS   = 180         # Downsample target sampling rate (ORIG_FS // 2)
SIGLEN      = 128         # Fixed length after downsampling (samples)
LEFT_PAD    = 40          # Left padding for R-peak window (in TARGET_FS)

TEST_RECS = {str(r) for r in [101,106,108,109,112,114,115,116,118,119,
                              122,124,201,203,205,207,208,209,215,220,223,230,234]}
MIT2AAMI  = {'N':0,'L':0,'R':0,'e':0,'j':0,
             'V':1,'E':1,
             'A':2,'a':2,'J':2,'S':2,
             'F':3,
             'Q':4,'|':4,'f':4}
AAMI      = {0:'N', 1:'V', 2:'S', 3:'F', 4:'Q'}
CLASSES   = len(AAMI)     # Fixed number of classes: 5
LEAD_N        = len(LEADS)          # 2
IN_CHANNELS   = LEAD_N * 2          # Time domain + frequency domain
AUG_FACTOR    = ORIG_FS / TARGET_FS # 2.0
LEFT_PAD_RAW  = int(LEFT_PAD * AUG_FACTOR)
SIGLEN_RAW    = int(SIGLEN * AUG_FACTOR)

# ===================== 1. Download ===================== #

def _list_record_headers() -> list[Path]:
    """Search for .hea files under the mitdb directory."""
    if not MITDB_DIR.exists():
        return []
    return sorted(MITDB_DIR.rglob("*.hea"))

def download_mitbih() -> None:
    """Ensure the 48 MIT-BIH records exist locally."""
    hea_files = _list_record_headers()
    need_redownload = (
        len(hea_files) < 48 or
        len(list(MITDB_DIR.rglob("*.dat"))) < 48 or
        len(list(MITDB_DIR.rglob("*.hea"))) < 48 or
        len(list(MITDB_DIR.rglob("*.atr"))) < 48
    )
    if not need_redownload:
        print("MIT-BIH already downloaded, skipping")
        return

    print(f"(Re) Downloading MIT-BIH → {RAW_DIR}")
    import shutil
    # Remove only the mitdb directory
    shutil.rmtree(MITDB_DIR, ignore_errors=True)
    MITDB_DIR.mkdir(parents=True, exist_ok=True)
    # WFDB will create RAW_DIR/mitdb/
    wfdb.dl_database("mitdb", dl_dir=str(RAW_DIR))
    print("Download complete")

# ===================== 2. Beat Segmentation ===================== #

def _filter(sig: np.ndarray, fs: int = ORIG_FS) -> np.ndarray:
    """0.5–40 Hz band-pass filter (neurokit2)."""
    return nk.signal_filter(sig, sampling_rate=fs, lowcut=0.5, highcut=40)

def _resample(sig: np.ndarray, desired_len: int = SIGLEN) -> np.ndarray:
    """Resample a 1D signal to the desired length."""
    return nk.signal_resample(sig, desired_length=desired_len)

def cut_beats() -> None:
    """Beat segmentation and normalization.

    To avoid data leakage, the StandardScaler is fitted on training records only.
    """
    if TRAIN_DIR.exists() and any(TRAIN_DIR.glob("*.npy")) and \
       TEST_DIR.exists() and any(TEST_DIR.glob("*.npy")) and \
       LABELS_PKL.exists():
        print(".npy files already exist, skipping cut_beats()")
        return

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True,  exist_ok=True)

    hea_files = _list_record_headers()
    if len(hea_files) < 48:
        raise FileNotFoundError("MIT-BIH records incomplete; please check the download directory")

    # Fit scaler using training records only
    train_files = [p for p in hea_files if p.stem not in TEST_RECS]
    scaler = StandardScaler()
    print("Pass-1: compute mean/std on TRAIN records")
    for hea in tqdm(train_files):
        rec_base = str(hea.with_suffix(""))  # Remove .hea
        sig, _ = wfdb.rdsamp(rec_base, channels=LEADS)
        sig = np.vstack([_filter(sig[:, i]) for i in range(LEAD_N)]).T  # (T, L)
        scaler.partial_fit(sig.reshape(-1, 1))

    # Segment and save beats
    print("Pass-2: segment beats and save .npy")
    labels = {}
    for hea in tqdm(hea_files):
        rec = hea.stem
        rec_base = str(hea.with_suffix(""))
        record = wfdb.rdrecord(rec_base, channels=LEADS)
        ann    = wfdb.rdann(rec_base, "atr")

        sig = record.p_signal.astype("float32")  # (T, L)
        sig = np.vstack([_filter(sig[:, i]) for i in range(LEAD_N)]).T
        sig = scaler.transform(sig.reshape(-1, 1)).reshape(-1, LEAD_N)  # Z-score

        for idx, (r, sym) in enumerate(zip(ann.sample, ann.symbol)):
            if sym not in MIT2AAMI:
                continue
            start, end = r - LEFT_PAD_RAW, r - LEFT_PAD_RAW + SIGLEN_RAW
            if start < 0 or end > len(sig):
                continue
            beat = sig[start:end]             # (SIGLEN_RAW, L)
            beat = beat.T                     # (L, SIGLEN_RAW)
            beat = np.stack([_resample(ch, SIGLEN) for ch in beat])  # (L, SIGLEN)
            fname   = f"{rec}_{idx:05d}.npy"
            target  = TEST_DIR if rec in TEST_RECS else TRAIN_DIR
            np.save(target / fname, beat.astype("float32"))
            labels[f"{rec}_{idx:05d}"] = MIT2AAMI[sym]

    with open(LABELS_PKL, "wb") as f:
        pickle.dump(labels, f)

    print(f"Beats saved: train={len(list(TRAIN_DIR.glob('*.npy')))}, "
          f"test={len(list(TEST_DIR.glob('*.npy')))}")

# ===================== 3. Dataset ===================== #
class ECGDataset(data.Dataset):
    """Return time-domain + frequency-domain channels for each beat."""
    def __init__(self, train: bool = True):
        self.dir   = TRAIN_DIR if train else TEST_DIR
        self.files = sorted(self.dir.glob("*.npy"))
        with open(LABELS_PKL, "rb") as f:
            self.labels = pickle.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp   = self.files[idx]
        x_td = np.load(fp)  # (L, SIGLEN)

        # Frequency domain: real-FFT magnitude + log1p compression
        x_fd = np.abs(np.fft.rfft(x_td, axis=1)).astype("float32")   # (L, SIGLEN//2+1)
        x_fd = np.log1p(x_fd)
        # Resample frequency feature length to match the time-domain length
        x_fd = np.stack([_resample(ch, SIGLEN) for ch in x_fd], axis=0)  # (L, SIGLEN)

        x    = np.concatenate([x_td, x_fd], axis=0)  # (L*2, SIGLEN)
        y    = int(self.labels[fp.stem])
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)

def load_data_ecg(batch_size: int = 256):
    """Create DataLoaders for training and testing."""
    train_set  = ECGDataset(True)
    test_set   = ECGDataset(False)

    g = torch.Generator(); g.manual_seed(SEED)
    pin = torch.cuda.is_available()

    train_iter = data.DataLoader(train_set, batch_size, shuffle=True,
                                 num_workers=0, drop_last=True,
                                 worker_init_fn=seed_worker, generator=g,
                                 pin_memory=pin)
    test_iter  = data.DataLoader(test_set,  batch_size, shuffle=False,
                                 num_workers=0,
                                 worker_init_fn=seed_worker, generator=g,
                                 pin_memory=pin)
    return train_iter, test_iter

# ===================== 4. Model ===================== #
class ECGNet(nn.Module):
    def __init__(self, in_channels: int = IN_CHANNELS, classes: int = 5, drop_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels,  32, 7, 2, 3), nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(drop_p),
            nn.Conv1d(32,  64, 5, 2, 2),          nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(drop_p),
            nn.Conv1d(64, 128, 3, 2, 1),          nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(drop_p),
            nn.Linear(128, classes)
        )

    def forward(self, x):
        return self.net(x)  # x: (B, C, T)

def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

# ===================== 5. Early-Stopping ===================== #
class EarlyStopping:
    """Stop training when the test metric fails to improve for `patience` epochs."""
    def __init__(self, patience: int = 10, verbose: bool = True):
        self.patience = patience
        self.best_acc = 0.0
        self.num_bad  = 0
        self.verbose  = verbose
        self.best_path = "best_model.pth"

    def __call__(self, acc: float, model: nn.Module):
        if acc > self.best_acc:
            self.best_acc = acc
            self.num_bad  = 0
            torch.save(model.state_dict(), self.best_path)
            if self.verbose:
                print(f"  new best test_acc={acc:.3f} — saved → {self.best_path}")
        else:
            self.num_bad += 1
            if self.verbose:
                print(f"  test_acc did not improve for {self.num_bad}/{self.patience} epochs")
        return self.num_bad >= self.patience

# ===================== 6. Train ===================== #

def accuracy(y_hat, y):
    return (y_hat.argmax(1) == y).sum()

def train(num_epochs: int = 15, batch: int = 1024, lr: float = 1e-3, device: str | None = None,
          drop_p: float = 0.3, weight_decay: float = 1e-4, patience_lr: int = 2, patience_es: int = 10):
    device = device or (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu")

    train_iter, test_iter = load_data_ecg(batch)
    C = CLASSES
    net = ECGNet(IN_CHANNELS, C, drop_p).to(device)
    net.apply(init_weights)

    loss_fn  = nn.CrossEntropyLoss()
    opt      = torch.optim.Adam(net.parameters(), lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                           patience=patience_lr, verbose=True)
    stopper   = EarlyStopping(patience_es)
    animator  = d2l.Animator(xlabel="epoch", ylabel="metric", xlim=[1, num_epochs],
                             legend=["train loss", "train acc", "test acc"])
    timer     = d2l.Timer()

    epochs_list, train_losses, train_accs, test_accs = [], [], [], []

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for X, y in train_iter:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_hat = net(X)
            l = loss_fn(y_hat, y)
            opt.zero_grad(); l.backward(); opt.step()
            metric.add(l.item() * y.numel(), accuracy(y_hat, y).item(), y.numel())

        train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
        net.eval(); test_metric = d2l.Accumulator(2)
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                test_metric.add(accuracy(net(X), y).item(), y.numel())
        test_acc = test_metric[0] / test_metric[1]
        scheduler.step(test_acc)

        epochs_list.append(epoch + 1)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        animator.add(epoch + 1, (train_loss, train_acc, test_acc))
        display.clear_output(wait=True)
        display.display(animator.fig)

        print(f"epoch {epoch + 1:2d}: loss {train_loss:.4f}  train_acc {train_acc:.3f}  test_acc {test_acc:.3f}")

        # Early stopping
        if stopper(test_acc, net):
            print("Early stopping triggered.")
            break

    animator.axes[0].set_title("Lightweight 1D CNN Result")
    elapsed = timer.stop()
    print(f"Time: {elapsed:.1f}s on {device}")
    print(f"Best test_acc={stopper.best_acc:.3f} — weights saved to {stopper.best_path}")

    # ===== Save CSV log =====
    with open("training_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "test_acc"])
        for e, tl, ta, va in zip(epochs_list, train_losses, train_accs, test_accs):
            w.writerow([e, tl, ta, va])
    print("Saved metrics → training_log.csv")

    # ===== Save static curve figure =====
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(epochs_list, train_losses, label="train loss")
    plt.plot(epochs_list, train_accs,  label="train acc")
    plt.plot(epochs_list, test_accs,   label="test acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved figure → training_curve.png")

    # ===================== Confusion matrix ===================== #
    print("\nGenerating confusion matrix for the best model...")
    net.load_state_dict(torch.load(stopper.best_path, map_location=device))
    net.eval()

    def _to_int_list(arr) -> list[int]:
        a = np.asarray(arr).reshape(-1)
        return [int(x) for x in a]

    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_iter:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            outputs = net(X)
            preds = outputs.argmax(1)
            # Force Python ints to avoid weird dtypes mixing in sets
            y_true.extend(_to_int_list(y.cpu().numpy()))
            y_pred.extend(_to_int_list(preds.cpu().numpy()))

    labels_used = sorted(set(y_true) | set(y_pred))
    valid_ids = set(AAMI.keys())
    unknown = [v for v in labels_used if v not in valid_ids]

    if unknown:
        from collections import Counter
        print("  Warning: unknown label ids detected in evaluation:", unknown)
        print("  y_true counts:", Counter(y_true))
        print("  y_pred counts:", Counter(y_pred))
        labels_for_cm = sorted(valid_ids & set(labels_used))
        if not labels_for_cm:
            labels_for_cm = sorted(valid_ids)
    else:
        labels_for_cm = labels_used

    cm = confusion_matrix(y_true, y_pred, labels=labels_for_cm)
    disp_labels = [AAMI.get(i, str(i)) for i in labels_for_cm]

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix - ECG Classification")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved figure → confusion_matrix.png")

# ===================== main ===================== #
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    download_mitbih()
    cut_beats()
    train()
