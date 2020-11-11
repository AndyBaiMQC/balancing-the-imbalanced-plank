'''
Test Module
'''

import time
import torch


def test_model(model,dataloaders,device,dataset_sizes, load_path='./saved_weight.pth'):

    print("#############################testing...")

    # ready to load
    model.load_state_dict(torch.load(load_path))
    since = time.time()

    for phase in ['test']:
        if phase == 'test':
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # these list to store the information
        # about true positive, false negtive and so on
        # in order to calculate the recall and presicion
        TP = [0, 0, 0]
        FN = [0, 0, 0]
        TN = [0, 0, 0]
        FP = [0, 0, 0]
        NM = [0, 0, 0]
        PR = [0, 0, 0]

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

                _, preds = torch.max(outputs, 1)

                for i in range(len(labels)):
                    NM[labels[i]] += 1
                    PR[preds[i]] += 1
                    if labels[i] == preds[i]:
                        TP[labels[i]] += 1
                        TN[(labels[i] - 1) % 3] += 1
                        TN[(labels[i] + 1) % 3] += 1

            # stats
            running_corrects += torch.sum(preds == labels.data)
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        for i in range(3):
            FP[i] = PR[i] - TP[i]
            FN[i] = NM[i] - TP[i]

        for i in range(3):
            print("metrics for label", i)
            print('Recall  :', (TP[i] + 0.0001) / (TP[i] + FN[i] + 0.0001))
            print('Precision  :', (TP[i] + +0.0001) / (TP[i] + FP[i] + 0.0001))

        print('{} Acc: {:.4f}'.format(phase, epoch_acc))

    time_elapsed = time.time() - since
    
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return