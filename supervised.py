from __future__ import division, print_function
import torchvision.datasets as dsets
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torchvision import models, transforms
import time
from torch.utils.tensorboard import SummaryWriter
import argparse

def main(args):
    num_classes = 2
    epochs = args.epochs

    # Define data transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create data loaders
    train_dataset = dsets.ImageFolder(args.train_data_dir, transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = dsets.ImageFolder(args.test_data_dir, transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define TensorBoard writer
    writer = SummaryWriter(args.log_dir)

    def train_model(model, criterion, optimizer, scheduler, num_epochs=epochs):
        since = time.time()
        i = 0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                running_corrects += (preds == labels).sum()

                running_train_acc = float((100 * running_corrects / running_loss))
                writer.add_scalar('Training Loss', loss, global_step=i)
                writer.add_scalar('Training Accuracy', running_train_acc, global_step=i)
                i += 1

            train_loss = running_loss / len(train_loader)
            print('Epoch {} \tTrain Loss: {:.3f}\tAccuracy: {:.3f}'.format(epoch, train_loss, 100 * running_corrects / running_loss))
            scheduler.step()

        writer.close()
        time_elapsed = time.time() - since

        print('Training finished in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        torch.save(model, args.model_path)

    def eval_model(model, criterion):
        correct = 0
        total = 0
        all_labels_np = np.empty([0, 1])
        all_predict_np = np.empty([0, 1]

        model.eval()

        with torch.no_grad():
            for data, target in test_loader:
                # Convert to numpy for Confusion Matrix
                labels_np = target.numpy()
                all_labels_np = np.concatenate((all_labels_np, labels_np), axis=None)

                data, target = data.to(device), target.to(device)

                outputs = model(data)
                _, pred = torch.max(outputs.data, 1)
                loss = criterion(outputs, target)

                total += target.size(0)
                correct += (pred == target).sum().item()

                # Confusion Matrix
                predicted_np = pred.cpu().detach().numpy()
                all_predict_np = np.concatenate((all_predict_np, predicted_np), axis=None)

                print('Test Loss: {:.3f}\tAccuracy: {:.3f}'.format(loss.data, 100 * correct / total))

        print(100 * correct / total)

        # Confusion Matrix
        all_labels_tensor = torch.from_numpy(all_labels_np)
        all_predict_tensor = torch.from_numpy(all_predict_np)
        all_labels_tensor = all_labels_tensor.type(torch.int64)
        all_predict_tensor = all_predict_tensor.type(torch.int64)

        stacked = torch.stack((all_labels_tensor, all_predict_tensor), dim=1)
        cmt = torch.zeros(2, 2, dtype=torch.int64)

        for p in stacked:
            tl, pl = p.tolist()
            cmt[tl, pl] = cmt[tl, pl] + 1
        print(cmt)

    model_ft = models.resnet101(pretrained=True)
    model_ft = model_ft.to(device)
    model_ft.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

    for name, child in model_ft.named_children():
        if name in ['fc']:
            print(name + ' is unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            print(name + ' is frozen')
            for param in child.parameters():
                param.requires_grad = False

    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=args.learning_rate, momentum=args.momentum)
    model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.step_size, gamma=args.gamma)

    train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epochs)
    eval_model(model_ft, criterion)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate your model.")
    parser.add_argument('--log_dir', type=str, default='runs/experiment', help='Directory for TensorBoard logs')
    parser.add_argument('--train_data_dir', type=str, default='./data/BUSI/cross/fold5/train/', help='Path to training data directory')
    parser.add_argument('--test_data_dir', type=str, default='./data/BUSI/cross/fold5/test/', help='Path to test data directory')
    parser.add_argument('--model_path', type=str, default='./eff.pt', help='Path to save the trained model')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.09, help='Momentum for optimizer')
    parser.add_argument('--step_size', type=int, default=4, help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for learning rate scheduler')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for data loader')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train')

    args = parser.parse_args()
    main(args)
