import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from copy import deepcopy

from config import Config, set_seed
from data import UnsupervisedTrainDataset, SupervisedDataset
from model import MultitaskUnsupervisedModel, SupervisedClassifier
from augment import feature_mask_aug, apply_gaussian_noise, apply_mixup_to_batch

def main(config: Config):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 20 + " STAGE 1: Multitask Unsupervised Pre-training " + "=" * 20)
    unsupervised_dataset = UnsupervisedTrainDataset(config.data_dir, config)
    unsupervised_dataloader = DataLoader(unsupervised_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    model = MultitaskUnsupervisedModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.unsupervised_lr, weight_decay=config.weight_decay)
    criterion_contrastive = nn.CrossEntropyLoss().to(device)
    criterion_recon = nn.MSELoss(reduction='mean')

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    for epoch in range(config.unsupervised_epochs):
        model.train()
        total_loss, total_c, total_r1, total_r2 = 0.0, 0.0, 0.0, 0.0
        ma_c, ma_r1, ma_r2 = 1.0, 1.0, 1.0
        eps = 1e-8

        for batch_data in unsupervised_dataloader:
            v_orig, s_orig, t_orig = [x.to(device) for x in batch_data]
            optimizer.zero_grad()

            (v_mixed, s_mixed, t_mixed), mixup_lam, mixup_indices = apply_mixup_to_batch((v_orig, s_orig, t_orig), config)

            (v_q, s_q, t_q), _, _ = feature_mask_aug(v_mixed, s_mixed, t_mixed, config)
            v_q, s_q, t_q = apply_gaussian_noise(v_q, s_q, t_q, noise_factor=0.05)

            (v_k, s_k, t_k), _, _ = feature_mask_aug(v_mixed, s_mixed, t_mixed, config)
            v_k, s_k, t_k = apply_gaussian_noise(v_k, s_k, t_k, noise_factor=0.05)

            logits_c, labels_c = model.forward_contrastive((v_q, s_q, t_q), (v_k, s_k, t_k))
            loss_c = criterion_contrastive(logits_c, labels_c)

            v_recon_clean, s_recon_clean, t_recon_clean = model.forward_reconstruction(v_q, s_q, t_q)
            loss_r2 = (nn.functional.mse_loss(v_recon_clean, v_orig) + nn.functional.mse_loss(s_recon_clean, s_orig) + nn.functional.mse_loss(t_recon_clean, t_orig)) / 3.0

            (v_mask_in, s_mask_in, t_mask_in), masked_modality, mask_indices = feature_mask_aug(v_orig, s_orig, t_orig, config)
            v_recon_mask, s_recon_mask, t_recon_mask = model.forward_reconstruction(v_mask_in, s_mask_in, t_mask_in)
            
            if masked_modality == 0:
                loss_r1 = nn.functional.mse_loss(v_recon_mask, v_orig)
            elif masked_modality == 1:
                loss_r1 = nn.functional.mse_loss(s_recon_mask, s_orig)
            else:
                loss_r1 = nn.functional.mse_loss(t_recon_mask, t_orig)

            if config.adaptive_loss_scaling:
                beta = config.als_beta
                ma_c  = beta * ma_c  + (1.0 - beta) * float(loss_c.detach().item())
                ma_r1 = beta * ma_r1 + (1.0 - beta) * float(loss_r1.detach().item())
                ma_r2 = beta * ma_r2 + (1.0 - beta) * float(loss_r2.detach().item())

                if epoch >= config.als_warmup_epochs:
                    tau = config.als_tau
                    ref = ma_r1
                    scale_c  = ((ref + eps) / (ma_c  + eps)) ** tau
                    scale_r2 = ((ref + eps) / (ma_r2 + eps)) ** tau

                    scale_c  = float(torch.clamp(torch.tensor(scale_c),  config.als_min_scale, config.als_max_scale))
                    scale_r2 = float(torch.clamp(torch.tensor(scale_r2),  config.als_min_scale, config.als_max_scale))
                else:
                    scale_c = scale_r2 = 1.0
            else:
                scale_c = scale_r2 = 1.0

            total_loss_step = (scale_c  * config.contrastive_weight  * loss_c +
                                config.recon_mask_weight   * loss_r1 +
                                scale_r2 * config.recon_clean_weight  * loss_r2)

            total_loss_step.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += total_loss_step.item()
            total_c += loss_c.item()
            total_r1 += loss_r1.item()
            total_r2 += loss_r2.item()

        avg_loss = total_loss / len(unsupervised_dataloader)
        avg_c = total_c / len(unsupervised_dataloader)
        avg_r1 = total_r1 / len(unsupervised_dataloader)
        avg_r2 = total_r2 / len(unsupervised_dataloader)
        print(f"Unsupervised Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f} (Contrastive: {avg_c:.4f}, Recon1_mask: {avg_r1:.4f}, Recon2_noise: {avg_r2:.4f})")

    trained_base_encoder = deepcopy(model.base_encoder_q)

    print("\n" + "=" * 20 + " Supervised Evaluation " + "=" * 20)
    train_dataset_sup = SupervisedDataset(config.data_dir, 'train', config)
    val_dataset_sup   = SupervisedDataset(config.data_dir, 'valid', config)
    test_dataset_sup  = SupervisedDataset(config.data_dir, 'test',  config)

    train_loader_sup = torch.utils.data.DataLoader(train_dataset_sup, batch_size=config.batch_size, shuffle=True)
    val_loader_sup   = torch.utils.data.DataLoader(val_dataset_sup,   batch_size=config.batch_size, shuffle=False)
    test_loader_sup  = torch.utils.data.DataLoader(test_dataset_sup,  batch_size=config.batch_size, shuffle=False)

    classifier_model = SupervisedClassifier(trained_base_encoder, config).to(device)
    optimizer_supervised = torch.optim.Adam(classifier_model.classifier_head.parameters(), lr=config.supervised_lr, weight_decay=config.weight_decay)
    criterion_supervised = nn.CrossEntropyLoss().to(device)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(config.supervised_epochs):
        classifier_model.train()
        train_loss = 0.0
        for v, s, t, labels in train_loader_sup:
            v, s, t, labels = v.to(device), s.to(device), t.to(device), labels.to(device)
            optimizer_supervised.zero_grad()
            outputs = classifier_model(v, s, t)
            loss = criterion_supervised(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier_model.classifier_head.parameters(), max_norm=1.0)
            optimizer_supervised.step()
            train_loss += loss.item()

        classifier_model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for v, s, t, labels in val_loader_sup:
                v, s, t, labels = v.to(device), s.to(device), t.to(device), labels.to(device)
                outputs = classifier_model(v, s, t)
                loss = criterion_supervised(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        avg_train_loss = train_loss / len(train_loader_sup)
        avg_val_loss = val_loss / len(val_loader_sup)

        print(f"Supervised Epoch {epoch + 1}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"  Val ACC: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = deepcopy(classifier_model.state_dict())
            print(f"  -> New best model found with validation Acc: {best_val_acc:.4f}")

    print("\n===== Final Evaluation on Test Set =====")
    if best_model_state:
        classifier_model.load_state_dict(best_model_state)
        print("Loaded best model from validation.")

    classifier_model.eval()
    all_preds_test, all_labels_test = [], []
    with torch.no_grad():
        for v, s, t, labels in test_loader_sup:
            outputs = classifier_model(v.to(device), s.to(device), t.to(device))
            preds = torch.argmax(outputs, dim=1)
            all_preds_test.extend(preds.cpu().numpy())
            all_labels_test.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels_test, all_preds_test)
    test_wf1 = f1_score(all_labels_test, all_preds_test, average='weighted', zero_division=0)
    test_mf1 = f1_score(all_labels_test, all_preds_test, average='macro', zero_division=0)
    test_balanced_acc = balanced_accuracy_score(all_labels_test, all_preds_test)

    print(f"\n--- Final Test Set Performance ---")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Balanced Accuracy: {test_balanced_acc:.4f}")
    print(f"Macro-F1: {test_mf1:.4f}")
    print(f"Weighted-F1: {test_wf1:.4f}")
    if config.n_classes == 2:
        test_bf1 = f1_score(all_labels_test, all_preds_test, average='binary', zero_division=0)
        print(f"Binary-F1: {test_bf1:.4f}")

if __name__ == '__main__':
    cfg = Config()
    main(cfg)
