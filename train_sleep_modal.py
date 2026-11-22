import modal

# 1. Define the cloud container environment & dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn", "mne")
)

app = modal.App("eeg-sleep-2ch", image=image)

# Attach the volume you uploaded in Step 2
vol = modal.Volume.from_name("sleep-eeg-vol")


# 2. Define the GPU training function (T4 is cheap and fast for 1D CNNs)
@app.function(
    gpu="T4",
    volumes={"/data": vol},
    timeout=3600,  # 1 hour max runtime
)
def train_model(num_files: int = 50, epochs: int = 15):
    import os
    import glob
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    import mne

    data_dir = "/data/sleep-cassette"
    psg_files = sorted(glob.glob(os.path.join(data_dir, "*PSG.edf")))
    hypnogram_files = sorted(glob.glob(os.path.join(data_dir, "*Hypnogram.edf")))

    print(f"Found {len(psg_files)} recordings. Loading {num_files}...")

    X_data, y_data = [], []
    fs = 100
    window_length = 3000
    eeg_channels = ["EEG Fpz-Cz", "EEG Pz-Oz"]
    stage_map = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 4,
        "Sleep stage R": 5,
    }

    for psg_file, hypnogram_file in zip(psg_files[:num_files], hypnogram_files[:num_files]):
        raw_edf = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
        eeg_2ch = raw_edf.get_data(picks=eeg_channels)
        annotations = mne.read_annotations(hypnogram_file)

        for onset, duration, description in zip(
            annotations.onset, annotations.duration, annotations.description
        ):
            if description not in stage_map:
                continue

            for segment in range(
                int(onset * fs), int(onset * fs + duration * fs), window_length
            ):
                end_sample = segment + window_length
                if end_sample <= eeg_2ch.shape[1]:
                    window = eeg_2ch[:, segment:end_sample]
                    if window.shape[1] == window_length:
                        X_data.append(window)
                        y_data.append(stage_map[description])

    X = np.array(X_data)
    y = np.array(y_data)

    # Channel-wise normalization
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True)
    X = (X - mean) / std

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} | Train shape: {X_train.shape}")

    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.LongTensor(y_test).to(device)

    model = nn.Sequential(
        nn.Conv1d(2, 32, kernel_size=50, stride=6),
        nn.ReLU(),
        nn.MaxPool1d(8),
        nn.Conv1d(32, 64, kernel_size=8),
        nn.ReLU(),
        nn.MaxPool1d(8),
        nn.Flatten(),
        nn.Linear(64 * 6, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 6),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch_size = 64

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(X_train):.5f}")

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test), dim=1).cpu().numpy()

    acc = accuracy_score(y_test.cpu().numpy(), preds)
    cm = confusion_matrix(y_test.cpu().numpy(), preds)

    # Save trained weights back to your Modal Volume
    torch.save(model.state_dict(), "/data/sleep_cnn_2ch.pt")
    vol.commit()

    return acc, cm


# 3. Local entrypoint: triggers the cloud function from your terminal
@app.local_entrypoint()
def main():
    acc, cm = train_model.remote(num_files=50, epochs=15)
    print(f"\nFinal Test Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)