"""
STUDENT IMPLEMENTATION REQUIRED

This file contains the training loop that you need to implement for HW1.
You should complete the train_model function by implementing the training logic
including optimizer setup, loss function, training loop, and model saving.

TODO: Implement the training loop in the train_model function
"""

# define your training loop here
import torch
from torch import nn
from torch import optim
from eval import evaluate_metrics

def train_model(model, predict_fn, train_loader, val_loader, device, save_path='best_model.pt'):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)#change lr, run main, eval

    num_epochs = 30
    patience = 5
    best_val_f1 = -1.0
    epochs_no_improve = 0

    hist = {"train_loss": [], "val_loss": []}

    print("Starting training...")
    print("="*60)

    for epoch in range(num_epochs):
        # TRAINING PHASE
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = train_loss /  max(n_batches, 1)
        
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():                    
            for vx, vy in val_loader:
                vx = vx.to(device).float()       
                vy = vy.to(device).float()             
                v_logits = model(vx)           
                v_loss = criterion(v_logits, vy)     
                val_loss += v_loss.item()           
                val_batches += 1               
        val_loss /= max(val_batches, 1)

         #VALIDATION PHASE
        with torch.no_grad():
            metrics = evaluate_metrics(model, val_loader, predict_fn, device)
            
        if isinstance(metrics, dict):
            val_f1 = metrics.get("weighted_f1", 0.0)
        else:
            val_f1 = metrics[0]
        
        hist["train_loss"].append(float(avg_train_loss))
        hist["val_loss"].append(float(val_loss))

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_train_loss:.4f}, Val F1: {val_f1:.4f}")
        # EARLY STOPPING
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best! F1: {val_f1:.4f}")
        else:
            epochs_no_improve += 1
        
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    #torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    final_metrics = evaluate_metrics(model, val_loader, predict_fn, device)
    print("="*60)
    print(f"*** Best (weighted) F1: {final_metrics['weighted_f1'] if isinstance(final_metrics, dict) else final_metrics[0]:.4f} ***")
    print(f'*** Best model weights saved at {save_path} ***')
    
    return model, hist
  