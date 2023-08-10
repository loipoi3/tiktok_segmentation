from torch.utils.data import DataLoader
import os
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss
from torch.utils.data import SubsetRandomSampler
import random
from dataset import TikTokDataset
import argparse
import config


def train_loop(model, criterion, optimizer, loader, scaler):
    model.train()

    losses, iou_scores, f1_scores, f2_scores, accuracys, recalls = [], [], [], [], [], []
    for idx, (img, mask) in enumerate(loader):
        # move data and targets to device(gpu or cpu)
        img = img.float().to(config.DEVICE)
        mask = mask.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            # making prediction
            pred = model(img)

            # calculate loss and dice coeficient and append it to losses and metrics
            Loss = criterion(pred, mask.float())

            tp, fp, fn, tn = smp.metrics.get_stats(torch.sigmoid(pred) > 0.5, mask.to(torch.int64), mode='binary',
                                                   num_classes=1)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
            recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
            print(Loss.item(), iou_score, f1_score, f2_score, accuracy, recall)
            losses.append(Loss.item())
            iou_scores.append(iou_score)
            f1_scores.append(f1_score)
            f2_scores.append(f2_score)
            accuracys.append(accuracy)
            recalls.append(recall)

        # backward
        optimizer.zero_grad()
        scaler.scale(Loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(f"train_loss: {sum(losses) / len(losses)}, iou_score: {sum(iou_scores) / len(iou_scores)}, "
          f"f1_score: {sum(f1_scores) / len(f1_scores)}, f2_score: {sum(f2_scores) / len(f2_scores)}, "
          f"accuracy: {sum(accuracys) / len(accuracys)}, recall:{sum(recalls) / len(recalls)}")


def val_loop(model, criterion, loader):
    model.eval()

    with torch.no_grad():
        losses, iou_scores, f1_scores, f2_scores, accuracys, recalls = [], [], [], [], [], []
        for idx, (img, mask) in enumerate(loader):
            # move data and targets to device(gpu or cpu)
            img = img.float().to(config.DEVICE)
            mask = mask.to(config.DEVICE)

            # making prediction
            pred = model(img)

            # calculate loss and dice coefficient and append it to losses and metrics
            Loss = criterion(pred, mask.float())

            tp, fp, fn, tn = smp.metrics.get_stats(torch.sigmoid(pred) > 0.5, mask.to(torch.int64), mode='binary',
                                                   num_classes=1)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
            recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

            losses.append(Loss.item())
            iou_scores.append(iou_score)
            f1_scores.append(f1_score)
            f2_scores.append(f2_score)
            accuracys.append(accuracy)
            recalls.append(recall)

        print(f"val_loss: {sum(losses) / len(losses)}, iou_score: {sum(iou_scores) / len(iou_scores)}, "
              f"f1_score: {sum(f1_scores) / len(f1_scores)}, f2_score: {sum(f2_scores) / len(f2_scores)}, "
              f"accuracy: {sum(accuracys) / len(accuracys)}, recall:{sum(recalls) / len(recalls)}")


def test_loop(model, criterion, loader):
    model.eval()

    with torch.no_grad():
        losses, iou_scores, f1_scores, f2_scores, accuracys, recalls = [], [], [], [], [], []
        for idx, (img, mask) in enumerate(loader):
            # move data and targets to device(gpu or cpu)
            img = img.float().to(config.DEVICE)
            mask = mask.to(config.DEVICE)

            # making prediction
            pred = model(img)

            # calculate loss and dice coefficient and append it to losses and metrics
            Loss = criterion(pred, mask.float())

            tp, fp, fn, tn = smp.metrics.get_stats(torch.sigmoid(pred) > 0.5, mask.to(torch.int64), mode='binary',
                                                   num_classes=1)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
            recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

            losses.append(Loss.item())
            iou_scores.append(iou_score)
            f1_scores.append(f1_score)
            f2_scores.append(f2_score)
            accuracys.append(accuracy)
            recalls.append(recall)
        print(f"test_loss: {sum(losses) / len(losses)}, iou_score: {sum(iou_scores) / len(iou_scores)}, "
              f"f1_score: {sum(f1_scores) / len(f1_scores)}, f2_score: {sum(f2_scores) / len(f2_scores)}, "
              f"accuracy: {sum(accuracys) / len(accuracys)}, recall:{sum(recalls) / len(recalls)}")


def main(args):
    root_dir_train = args["root_dir_train"]
    root_dir_val = args["root_dir_val"]
    root_dir_test = args["root_dir_test"]
    saved_model_path = args["save_model_path"]
    num_epochs = args["epochs"]
    learning_rate = args["lr"]
    batch_size = args["batch"]

    # if directory model exists than create this
    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)

    # define model and move it to device(gpu or cpu)
    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
        activation=None
    )
    model.to(config.DEVICE)

    train_dataset = TikTokDataset(root_dir=root_dir_train, transform=config.TRANSFORM_TRAIN)
    val_dataset = TikTokDataset(root_dir=root_dir_val, transform=config.TRANSFORM_VAL_TEST)
    test_dataset = TikTokDataset(root_dir=root_dir_test, transform=config.TRANSFORM_VAL_TEST)

    # Get the total number of samples in each dataset
    num_train_samples = len(train_dataset)
    num_val_samples = len(val_dataset)
    num_test_samples = len(test_dataset)

    # Create a list of indices representing the entire dataset
    all_indices_train = list(range(num_train_samples))
    all_indices_val = list(range(num_val_samples))
    all_indices_test = list(range(num_test_samples))

    # checking whether the model needs to be retrained
    if config.LOAD_MODEL:
        model = smp.Unet(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
            activation=None
        )
        model.to(config.DEVICE)
        model.load_state_dict(torch.load(config.PATH_TO_MODEL))

    # define loss function
    criterion = SoftBCEWithLogitsLoss()

    # Create optimizer only for trainable parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # define scaler
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        # Randomly shuffle the list of indices
        random.shuffle(all_indices_train)
        random.shuffle(all_indices_val)
        random.shuffle(all_indices_test)

        # Split indices for train, validation, and test sets
        train_indices = all_indices_train[:5000]
        val_indices = all_indices_val[:1000]
        test_indices = all_indices_test[:1000]

        # Use SubsetRandomSampler to create DataLoader with fixed subsets for train, validation, and test
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        # Create the DataLoader objects for each dataset to enable easy batching during training and evaluation.
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, sampler=val_sampler)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=test_sampler)

        print(f'Epoch: {epoch + 1}')

        train_loop(model, criterion, optimizer, train_loader, scaler)
        val_loop(model, criterion, val_loader)
        test_loop(model, criterion, test_loader)

        # save model
        torch.save(model.state_dict(), saved_model_path + f'model{epoch + 1}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir_train", type=str, default=config.ROOT_DIR_TRAIN,
                        help='Specify path to your images and masks of train dataset')
    parser.add_argument("--root_dir_val", type=str, default=config.ROOT_DIR_VAL,
                        help='Specify path to your images and masks of validation dataset')
    parser.add_argument("--root_dir_test", type=str, default=config.ROOT_DIR_TEST,
                        help='Specify path to your images and masks of test dataset')
    parser.add_argument("--save_model_path", type=str, default=config.SAVED_MODEL_PATH,
                        help='Specify path for save models, where models folder will be created')
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help='Specify epoch for model training')
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help='Specify learning rate')
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE, help='Specify batch size')
    args = parser.parse_args()
    args = vars(args)
    main(args)
