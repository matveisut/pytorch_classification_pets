import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    batch_size = dataloader.batch_size
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)

        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if batch % 20 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
       


def test_loop(dataloader, model, loss_fn):

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    
    all_y_true = []
    all_y_pred = []
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            
            
            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            correct += (pred.argmax(1) == y ).type(torch.float).sum().item()
            
            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(pred.argmax(1).cpu().numpy())


        
    all_y_pred = np.array(all_y_pred)
    all_y_true = np.array(all_y_true)
    
    # Weighted averages (усреднение с весами по поддержке классов)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average='weighted', zero_division=0
    )
    
    test_loss /= num_batches
    correct /= size
    accuracy = correct
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    print("weighted averages")
    print(f"Precision: {precision_weighted:.4f}")
    print(f"Recall:    {recall_weighted:.4f}")
    print(f"F1:        {f1_weighted:.4f}")
    
    return {
        'accuracy': accuracy,
        'loss': test_loss,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }
