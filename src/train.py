'''
Train Module
'''

import time
import torch
import matplotlib.pyplot as plt


def train_model(model, criterion, optimizer, num_epochs,dataloaders, device,dataset_sizes,save_path='./saved_weight.pth'):
    
    print("#############################training...")
    since = time.time()
    loss_epoch = []

    for epoch in range(num_epochs):
        TP = [0, 0, 0]
        FN = [0, 0, 0]
        TN = [0, 0, 0]
        FP = [0, 0, 0]
        NM = [0, 0, 0]
        PR = [0, 0, 0]
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                for i in range(len(labels)):
                    NM[labels[i]] += 1
                    PR[preds[i]] += 1
                    if labels[i] == preds[i]:
                        TP[labels[i]] += 1
                        TN[(labels[i] - 1) % 3] += 1
                        TN[(labels[i] + 1) % 3] += 1
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        for i in range(3):
            FP[i] = PR[i] - TP[i]
            FN[i] = NM[i] - TP[i]

        for i in range(3):
            print("metrics for label", i)
            print('Recall  :', (TP[i] + 0.0001) / (TP[i] + FN[i] + 0.0001))
            print('Precision  :', (TP[i] + +0.0001) / (TP[i] + FP[i] + 0.0001))

        epoch_loss = running_loss / dataset_sizes['train']
        loss_epoch.append(epoch_loss)
        epoch_acc = running_corrects.double() / dataset_sizes['train']
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))

    plt.plot(range(len(loss_epoch)), loss_epoch)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    torch.save(model.state_dict(), save_path)

    return model
    