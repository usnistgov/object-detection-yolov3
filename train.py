import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import os
# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# gpus_to_use must bs comma separated list of gpu ids, e.g. "1,3,4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1" for multiple

import os
# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# gpus_to_use must bs comma separated list of gpu ids, e.g. "1,3,4"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # "0, 1" for multiple

import numpy as np
import torch.utils.data
import json
import csv

import yolo_dataset
import model


def plot(train_loss, test_loss, name, output_folder=None):
    mpl.rcParams['agg.path.chunksize'] = 10000  # fix for error in plotting large numbers of points

    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = plt.gca()

    plt.title("Test Loss {} vs. Number of Training Epochs".format(name))
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    train_scale_factor = float(len(test_loss)) / float(len(train_loss))
    x_vals = np.asarray(list(range(len(train_loss)))) * train_scale_factor
    y_vals = np.asarray(train_loss)
    ax.scatter(x_vals, y_vals, c='b', s=1, label="Train")
    x_vals = np.asarray(list(range(len(test_loss)))) + 1
    y_vals = np.asarray(test_loss)
    ax.plot(x_vals, y_vals, 'r-', marker='o', markersize=4, label="Test", linewidth=1)
    ax.set_yscale('log')
    nbs = list()
    nbs.extend(train_loss)
    nbs.extend(test_loss)
    nbs = np.asarray(nbs)
    p99 = np.percentile(nbs, 99)
    idx = np.nonzero(nbs)[0]
    if len(idx) > 0:
        mv = np.min(nbs[np.nonzero(nbs)[0]])
    else:
        mv = 1e-8
    plt.ylim((mv, p99))
    plt.legend()
    fn = '{}.png'.format(name)
    if output_folder is not None:
        fig.savefig(os.path.join(output_folder, fn))
    else:
        fig.savefig(fn)
    plt.close(fig)


def train_model(config, output_folder, early_stopping_count):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    use_gpu = torch.cuda.is_available()
    num_workers = 10 #int(config['batch_size'] / 2)

    torch_model_ofp = os.path.join(output_folder, 'checkpoint')
    if os.path.exists(torch_model_ofp):
        import shutil
        shutil.rmtree(torch_model_ofp)
    os.makedirs(torch_model_ofp)

    pin_dataloader_memory = True

    train_dataset = yolo_dataset.YoloDataset(config['train_lmdb_filepath'], augment=config['augment'])
    train_sampler = train_dataset.get_weighted_sampler()  # this tells pytorch how to weight samples to balance the classes
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=num_workers, sampler=train_sampler, pin_memory=pin_dataloader_memory, drop_last=True)

    test_dataset = yolo_dataset.YoloDataset(config['test_lmdb_filepath'], augment=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=num_workers, pin_memory=pin_dataloader_memory, drop_last=True)

    config['number_classes'] = train_dataset.get_number_classes()
    config['image_size'] = train_dataset.get_image_shape()
    config['weight_decay'] = 5e-4

    config_ofp = os.path.join(torch_model_ofp, 'config.json')
    with open(config_ofp, 'w') as fp:
        json.dump(config, fp)
    yolo_model = model.YoloV3(config)
    if use_gpu:
        yolo_model = torch.nn.DataParallel(yolo_model)
        # move model to GPU
        yolo_model = yolo_model.cuda()

    optimizer = torch.optim.Adam(yolo_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    loss_keys = ['total', 'xy', 'wh', 'obj', 'cls']
    train_loss = list()
    train_loss_dict = dict()
    for name in loss_keys:
        train_loss_dict[name] = list()


    test_loss_dict = dict()
    for name in loss_keys:
        test_loss_dict[name] = list()
    epoch = 0
    while True:
        print('-' * 10)
        print('Epoch {}'.format(epoch))

        print('running training epoch')
        yolo_model.train()  # put the model in training mode
        batch_count = 0
        for i, (images, target) in enumerate(train_loader):
            if use_gpu:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            optimizer.zero_grad()
            batch_count = batch_count + 1

            # compute output
            loss = yolo_model(images, target)
            # loss is [1, 5]

            for l_idx in range(len(loss_keys)):
                train_loss_dict[loss_keys[l_idx]].append(float(loss[0, l_idx].detach().cpu().numpy()))
            train_loss.append(train_loss_dict[loss_keys[0]])
            print("Epoch: {} Batch {}/{} loss {}".format(epoch, i, len(train_loader), train_loss_dict[loss_keys[0]][-1]))
            if np.isnan(train_loss_dict[loss_keys[0]][-1]):
                raise RuntimeError("loss went to NaN")

            loss = loss[0, 0]

            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()

        for key in loss_keys:
            print('  loss {} = {}'.format(key, np.average(train_loss_dict[key][-batch_count:])))

        print('running test epoch')
        yolo_model.eval()
        epoch_test_loss_dict = dict()
        for name in loss_keys:
            epoch_test_loss_dict[name] = list()
        with torch.no_grad():
            for i, (images, target) in enumerate(test_loader):
                if use_gpu:
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                optimizer.zero_grad()

                # compute output
                loss = yolo_model(images, target)
                # loss is [1, 5]

                for l_idx in range(len(loss_keys)):
                    epoch_test_loss_dict[loss_keys[l_idx]].append(float(loss[0, l_idx].detach().cpu().numpy()))

        for key in loss_keys:
            test_loss_dict[key].append(np.average(epoch_test_loss_dict[key]))

        print('Test Loss')
        for key in train_loss_dict.keys():
            print('  loss {} = {}'.format(key, test_loss_dict[key][-1]))

        for key in loss_keys:
            plot(train_loss_dict[key], test_loss_dict[key], "{}-loss".format(key), output_folder)
            with open(os.path.join(output_folder, 'train-{}-loss.csv'.format(key)), 'w') as file:
                val = train_loss_dict[key]
                for i in range(len(val)):
                    file.write('{}\n'.format(val[i]))
            with open(os.path.join(output_folder, 'test-{}-loss.csv'.format(key)), 'w') as file:
                val = test_loss_dict[key]
                for i in range(len(val)):
                    file.write('{}\n'.format(val[i]))

        CONVERGENCE_TOLERANCE = 1e-4
        print('Best Current Epoch Selection:')
        print('Test Loss:')
        test_loss = test_loss_dict[loss_keys[0]]
        print(test_loss)
        min_test_loss = np.min(test_loss)
        error_from_best = np.abs(test_loss - min_test_loss)
        error_from_best[error_from_best < CONVERGENCE_TOLERANCE] = 0
        best_epoch = np.where(error_from_best == 0)[0][0]  # unpack numpy array, select first time since that value has happened
        print('Best epoch: {}'.format(best_epoch))

        # determine if to record a new checkpoint based on best test loss
        if (len(test_loss) - 1) == np.argmin(test_loss):
            torch.save({
                'epoch': epoch,
                'model_state_dict': yolo_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(torch_model_ofp, 'yolov3.ckpt'))

        if len(test_loss) - best_epoch > early_stopping_count:
            break  # break the epoch loop
        epoch = epoch + 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog='train', description='Script which trains a yolo_v3 model')

    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        help='training batch size', default=8)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_database', dest='train_database_filepath', type=str,
                        help='lmdb database to use for (Required)', required=True)
    parser.add_argument('--test_database', dest='test_database_filepath', type=str,
                        help='lmdb database to use for testing (Required)', required=True)
    parser.add_argument('--output_dir', dest='output_folder', type=str,
                        help='Folder where outputs will be saved (Required)', required=True)
    parser.add_argument('--early_stopping', dest='early_stopping', type=int, help='Perform early stopping when the test loss does not improve for N epochs.', default=10)
    parser.add_argument('--use_augmentation', dest='use_augmentation', type=int,
                        help='whether to use data augmentation [0 = false, 1 = true]', default=1)

    args = parser.parse_args()

    batch_size = args.batch_size
    train_database_filepath = args.train_database_filepath
    test_database_filepath = args.test_database_filepath
    output_folder = args.output_folder
    early_stopping = args.early_stopping
    learning_rate = args.learning_rate
    use_augmentation = args.use_augmentation

    print('Arguments:')
    print('batch_size = {}'.format(batch_size))
    print('train_database_filepath = {}'.format(train_database_filepath))
    print('test_database_filepath = {}'.format(test_database_filepath))
    print('output folder = {}'.format(output_folder))
    print('early_stopping = {}'.format(early_stopping))
    print('learning_rate = {}'.format(learning_rate))
    print('use_augmentation = {}'.format(use_augmentation))

    config = dict()
    config['anchors'] = [[75, 75], [75, 75], [75, 75]]
    # config['anchors'] = [(32, 32), (128, 128), (256, 256)]
    config['anchors_mask'] = [[0], [0], [0]]
    config['batch_size'] = batch_size
    config['learning_rate'] = learning_rate
    config['train_lmdb_filepath'] = train_database_filepath
    config['test_lmdb_filepath'] = test_database_filepath
    config['augment'] = use_augmentation

    train_model(config, output_folder, early_stopping)
