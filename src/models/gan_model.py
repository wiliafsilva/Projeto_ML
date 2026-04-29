import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math


class MLPGenerator(nn.Module):
    def __init__(self, noise_dim, label_dim, output_dim, hidden_sizes=(128, 128)):
        super().__init__()
        input_dim = noise_dim + label_dim
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        # For tabular data we don't force activation; let outputs be raw and rely on training
        self.net = nn.Sequential(*layers)

    def forward(self, z, labels_onehot):
        x = torch.cat([z, labels_onehot], dim=1)
        return self.net(x)


class MLPDiscriminator(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_sizes=(128, 128)):
        super().__init__()
        # Discriminator is conditional: receives features and label information
        self.input_dim = input_dim
        in_dim = input_dim + label_dim
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LeakyReLU(0.2))
            prev = h
        # Output for real/fake
        self.features = nn.Sequential(*layers)
        self.output_rf = nn.Linear(prev, 1)  # real/fake logit
        # Output for class prediction (multi-class)
        self.output_clf = nn.Linear(prev, label_dim)

    def forward(self, x, labels_onehot):
        # x: features, labels_onehot: conditional information
        inp = torch.cat([x, labels_onehot], dim=1)
        h = self.features(inp)
        rf_logit = self.output_rf(h)
        class_logit = self.output_clf(h)
        return rf_logit, class_logit


class GANClassifier:
    """
    Minimal conditional GAN wrapper where the Discriminator is used as a classifier.

    - fit(X, y): trains a cGAN adversarially. Sample weights are ignored (documented).
    - predict(X): uses discriminator's class logits to pick argmax.
    - predict_proba(X): returns softmax probabilities from discriminator.

    Notes:
    - Designed to be small and stable with conservative defaults.
    - Uses PyTorch internally but the wrapper is picklable via storing state_dicts and params.
    """

    def __init__(self,
                 epochs=50,
                 batch_size=64,
                 lr=1e-3,
                 noise_dim=32,
                 gen_hidden=(128, 128),
                 disc_hidden=(128, 128),
                 device=None,
                 random_state=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.noise_dim = noise_dim
        self.gen_hidden = gen_hidden
        self.disc_hidden = disc_hidden
        self.random_state = random_state

        # Device selection: GPU if available and not explicitly forced otherwise
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # These will be created at fit time when input dims are known
        self.generator = None
        self.discriminator = None
        self.label_dim = None
        self.input_dim = None

    def _set_seed(self):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

    def fit(self, X, y, sample_weight=None):
        """
        Train the cGAN. sample_weight is ignored (explicitly) to keep interface compatible.
        """
        if sample_weight is not None:
            print("[GANClassifier] Aviso: sample_weight fornecido será ignorado pelo GAN.")

        self._set_seed()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        n_samples, n_features = X.shape
        classes = np.unique(y)
        self.label_dim = int(classes.max() + 1)  # assume labels are 0..K-1
        self.input_dim = n_features

        # Build models
        self.generator = MLPGenerator(self.noise_dim, self.label_dim, n_features, hidden_sizes=self.gen_hidden).to(self.device)
        self.discriminator = MLPDiscriminator(n_features, self.label_dim, hidden_sizes=self.disc_hidden).to(self.device)

        # Optimizers
        optim_g = optim.Adam(self.generator.parameters(), lr=self.lr)
        optim_d = optim.Adam(self.discriminator.parameters(), lr=self.lr)

        # Losses
        bce = nn.BCEWithLogitsLoss()
        ce = nn.CrossEntropyLoss()

        # Prepare dataset
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.epochs + 1):
            self.discriminator.train()
            self.generator.train()
            epoch_loss_d = 0.0
            epoch_loss_g = 0.0

            for real_x, real_y in loader:
                real_x = real_x.to(self.device)
                real_y = real_y.to(self.device)

                batch_size = real_x.size(0)

                # one-hot for labels
                labels_onehot = torch.nn.functional.one_hot(real_y, num_classes=self.label_dim).float().to(self.device)

                # ---------------------
                # Train Discriminator
                # ---------------------
                optim_d.zero_grad()

                # Real samples
                rf_logit_real, class_logit_real = self.discriminator(real_x, labels_onehot)
                loss_real_rf = bce(rf_logit_real.view(-1), torch.ones(batch_size, device=self.device))
                loss_real_clf = ce(class_logit_real, real_y)

                # Fake samples
                z = torch.randn(batch_size, self.noise_dim, device=self.device)
                # sample labels for fake equal to real labels (conditional)
                z_labels = labels_onehot
                fake_x = self.generator(z, z_labels).detach()
                rf_logit_fake, class_logit_fake = self.discriminator(fake_x, z_labels)
                loss_fake_rf = bce(rf_logit_fake.view(-1), torch.zeros(batch_size, device=self.device))
                # For fake samples we don't compute classification loss (or could use uniform)

                loss_d = loss_real_rf + loss_fake_rf + loss_real_clf * 0.5  # weight classification modestly
                loss_d.backward()
                optim_d.step()

                # ---------------------
                # Train Generator
                # ---------------------
                optim_g.zero_grad()
                z = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_x = self.generator(z, z_labels)
                rf_logit_fake_g, class_logit_fake_g = self.discriminator(fake_x, z_labels)

                # Generator wants discriminator to predict real (1) and correct class
                loss_g_rf = bce(rf_logit_fake_g.view(-1), torch.ones(batch_size, device=self.device))
                # Encourage class consistency: target is the conditional label
                loss_g_clf = ce(class_logit_fake_g, real_y)

                loss_g = loss_g_rf + 0.5 * loss_g_clf
                loss_g.backward()
                optim_g.step()

                epoch_loss_d += loss_d.item()
                epoch_loss_g += loss_g.item()

            # modest logging per epoch
            if epoch % max(1, self.epochs // 10) == 0 or epoch == 1 or epoch == self.epochs:
                avg_d = epoch_loss_d / len(loader)
                avg_g = epoch_loss_g / len(loader)
                print(f"[GAN] Epoch {epoch}/{self.epochs} - D_loss: {avg_d:.4f} | G_loss: {avg_g:.4f}")

        # After training keep models in eval mode
        self.discriminator.eval()
        self.generator.eval()

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        xs = torch.from_numpy(X).to(self.device)
        # Dummy labels for conditioning: use zeros (not used by discriminator for predicting classes)
        # But we need labels_onehot as input; use uniform zeros which is acceptable since discriminator uses
        # features + label; to get unconditional class probabilities we can feed zero vector and rely on
        # class_head output.
        batch_size = xs.size(0)
        labels_onehot = torch.zeros(batch_size, self.label_dim, device=self.device)
        with torch.no_grad():
            _, class_logit = self.discriminator(xs, labels_onehot)
            probs = torch.softmax(class_logit, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    # Make object picklable by saving parameters and state_dicts
    def __getstate__(self):
        state = self.__dict__.copy()
        # Replace torch modules with their state_dicts
        if self.generator is not None:
            state['generator_state'] = self.generator.state_dict()
        else:
            state['generator_state'] = None
        if self.discriminator is not None:
            state['discriminator_state'] = self.discriminator.state_dict()
        else:
            state['discriminator_state'] = None

        # Remove actual module objects (not picklable across devices reliably)
        state['generator'] = None
        state['discriminator'] = None
        # device is not picklable directly; store as string
        state['device'] = str(self.device)
        return state

    def __setstate__(self, state):
        # Restore basic attributes
        device = torch.device(state.get('device', 'cpu'))
        self.__dict__.update(state)
        self.device = device

        # Rebuild modules if we have dimensions
        if self.input_dim is not None and self.label_dim is not None:
            self.generator = MLPGenerator(self.noise_dim, self.label_dim, self.input_dim, hidden_sizes=self.gen_hidden).to(self.device)
            self.discriminator = MLPDiscriminator(self.input_dim, self.label_dim, hidden_sizes=self.disc_hidden).to(self.device)
            if state.get('generator_state') is not None:
                self.generator.load_state_dict(state['generator_state'])
            if state.get('discriminator_state') is not None:
                self.discriminator.load_state_dict(state['discriminator_state'])
