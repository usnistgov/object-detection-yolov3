import sys

if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import os

# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import numpy as np
import torch
import torch.utils.data
import json
import torch.cuda.amp

import yolo_layer
import yolo_dataset
import model
import utils
import compute_confusion_matrix


def plot_train_test(train_loss, test_loss, name, output_folder=None):
    mpl.rcParams['agg.path.chunksize'] = 10000  # fix for error in plotting large numbers of points

    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = plt.gca()

    plt.title("Test Loss {} vs. Number of Training Epochs".format(name))
    plt.xlabel("Training Epochs")
    plt.ylabel("{} Loss".format(str.capitalize(name)))
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
    nbs = nbs[np.isfinite(nbs)]
    p99 = np.percentile(nbs, 99)
    idx = np.isfinite(nbs)
    nbs = nbs[idx]
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


def train_model(config, output_folder, early_stopping_count, use_amp):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('Found {} gpus'.format(torch.cuda.device_count()))

    if use_amp:
        print('Using AMP')

    num_workers = 30  # TODO remove

    torch_model_ofp = os.path.join(output_folder, 'checkpoint')
    if os.path.exists(torch_model_ofp):
        import shutil
        shutil.rmtree(torch_model_ofp)
    os.makedirs(torch_model_ofp)

    pin_dataloader_memory = True
    torch.backends.cudnn.benchmark = True  # autotune cudnn kernel choice
    # disable debugging API, turn on for debugging
    torch.autograd.set_detect_anomaly(False)

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
        json.dump(config, fp, indent=2)
    yolo_model = model.YoloV3(config)
    yolo_model = torch.nn.DataParallel(yolo_model)
    # TODO Setup Distributed Data parallel at a later date
    if use_gpu:
        print('Moving model to the GPU')
        # move model to GPU
        yolo_model.cuda()
        print('Found {} gpus'.format(torch.cuda.device_count()))

    optimizer = torch.optim.Adam(yolo_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # use step size 1, and only call scheduler.step() when I want to
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # # load a checkpoint to continue refining training
    # checkpoint_filepath = ''
    # if checkpoint_filepath is not None:
    #     checkpoint = torch.load(checkpoint_filepath)
    #     yolo_model.load_state_dict(checkpoint['model_state_dict'])
    #     yolo_model = yolo_model.cuda()
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     # set the learning rate
    #     for g in optimizer.param_groups:
    #         g['lr'] = 1e-5

    loss_keys = ['total', 'xy', 'wh', 'obj', 'cls']
    train_loss = list()
    train_loss_dict = dict()
    for name in loss_keys:
        train_loss_dict[name] = list()

    test_loss_dict = dict()
    for name in loss_keys:
        test_loss_dict[name] = list()
    TP_count_list = list()
    FP_count_list = list()
    FN_count_list = list()
    epoch = 0
    LR_change_epoch = 0
    nb_learning_rate_reductions = 0
    CONVERGENCE_TOLERANCE = 0.1

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    done_training = False
    while not done_training:
        print('-' * 10)
        print('Epoch {}'.format(epoch))

        print('running training epoch')
        yolo_model.train()  # put the model in training mode
        batch_count = 0
        for i, (images, target) in enumerate(train_loader):
            if use_gpu:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # optimizer.zero_grad()
            # faster zero_grad
            for param in yolo_model.parameters():
                param.grad = None
            batch_count = batch_count + 1

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = yolo_model(images, target)
            else:
                loss = yolo_model(images, target)
            # loss = yolo_model(images, target)
            # loss is [1, 5]
            total_loss = loss[0, 0]

            if use_amp:
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(total_loss).backward()
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
                if not np.isnan(total_loss.detach().cpu().numpy()):
                    # don't record the loss if it went to nan or inf, since that means the scaler was updated and the batch ignored
                    for l_idx in range(len(loss_keys)):
                        train_loss_dict[loss_keys[l_idx]].append(float(loss[0, l_idx].detach().cpu().numpy()))
                    train_loss.append(train_loss_dict[loss_keys[0]])
            else:
                # compute gradient and do SGD step
                total_loss.backward()
                optimizer.step()

                for l_idx in range(len(loss_keys)):
                    train_loss_dict[loss_keys[l_idx]].append(float(loss[0, l_idx].detach().cpu().numpy()))
                train_loss.append(train_loss_dict[loss_keys[0]])
                if np.isnan(train_loss_dict[loss_keys[0]][-1]):
                    raise RuntimeError("loss went to NaN")

            if i % 100 == 0 and len(train_loss_dict[loss_keys[0]]) > 0:
                print("Epoch: {} Batch {}/{} loss {}".format(epoch, i, len(train_loader), train_loss_dict[loss_keys[0]][-1]))

        print('Train Loss Epoch {}'.format(epoch))
        for key in loss_keys:
            print('  loss {} = {}'.format(key, np.average(train_loss_dict[key][-batch_count:])))

        print('running test epoch')
        yolo_model.eval()
        epoch_test_loss_dict = dict()
        for name in loss_keys:
            epoch_test_loss_dict[name] = list()
        TP_count = 0
        FP_count = 0
        FN_count = 0
        with torch.no_grad():
            for i, (images, target) in enumerate(test_loader):
                if use_gpu:
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                optimizer.zero_grad()

                if use_amp:
                    with torch.cuda.amp.autocast():
                        loss = yolo_model(images, target)
                else:
                    loss = yolo_model(images, target)
                # loss is [1, 5]

                for l_idx in range(len(loss_keys)):
                    epoch_test_loss_dict[loss_keys[l_idx]].append(float(loss[0, l_idx].detach().cpu().numpy()))

                if config['generate_confusion']:
                    MIN_SCORE_THRESHOLD = 0.1
                    IOU_THRESHOLD = 0.3
                    min_box_size = 12
                    feature_maps = yolo_model(images)

                    predictions = list()
                    for i in range(len(feature_maps)):
                        feature_map = feature_maps[i].detach().cpu().numpy()
                        preds = utils.reorg_layer_np(feature_map, yolo_layer.YOLOLayer.STRIDES[i], config['number_classes'], config['anchors'])
                        predictions.append(preds)
                    predictions = np.concatenate(predictions, axis=1)

                    predictions = utils.postprocess_numpy(predictions, yolo_model.module.number_classes,
                                                          score_threshold=MIN_SCORE_THRESHOLD, iou_threshold=IOU_THRESHOLD,
                                                          min_box_size=min_box_size)
                    # predictions = [x, y, w, h, score, pred_class] where (x, y) is upper left
                    target = target.detach().cpu().numpy()

                    # loop over the batches since they must be handled as individual images
                    for b in range(target.shape[0]):
                        tgt = target[b]
                        pred = predictions[b]
                        if pred is None:
                            # no boxes detected
                            pred = np.zeros((0, 6))
                        idx = np.sum(tgt, axis=1) > 0
                        tgt = tgt[idx, :]
                        # convert [x, y, w, h] to [x1, y1, x2, y2]
                        tgt[:, 2] = tgt[:, 0] + tgt[:, 2]
                        tgt[:, 3] = tgt[:, 1] + tgt[:, 3]
                        pred[:, 2] = pred[:, 0] + pred[:, 2]
                        pred[:, 3] = pred[:, 1] + pred[:, 3]
                        TP_list, FP_list, FN_list = compute_confusion_matrix.confusion_matrix(tgt, pred)
                        TP_count += len(TP_list)
                        FP_count += len(FP_list)
                        FN_count += len(FN_list)

        for key in loss_keys:
            test_loss_dict[key].append(np.average(epoch_test_loss_dict[key]))

        print('Test Loss Epoch {}'.format(epoch))
        for key in train_loss_dict.keys():
            print('  loss {} = {}'.format(key, test_loss_dict[key][-1]))
        if config['generate_confusion']:
            TP_count_list.append(TP_count)
            FP_count_list.append(FP_count)
            FN_count_list.append(FN_count)
            print('  TP = {}'.format(TP_count))
            print('  FP = {}'.format(FP_count))
            print('  FN = {}'.format(FN_count))

        for key in loss_keys:
            plot_train_test(train_loss_dict[key], test_loss_dict[key], "{}-loss".format(key), output_folder)
            with open(os.path.join(output_folder, 'train-{}-loss.csv'.format(key)), 'w') as file:
                val = train_loss_dict[key]
                for i in range(len(val)):
                    file.write('{}\n'.format(val[i]))
            with open(os.path.join(output_folder, 'test-{}-loss.csv'.format(key)), 'w') as file:
                val = test_loss_dict[key]
                for i in range(len(val)):
                    file.write('{}\n'.format(val[i]))

        if config['generate_confusion']:
            with open(os.path.join(output_folder, 'confusion.csv'), 'w') as file:
                file.write('Epoch, TP, FP, FN\n')
                for i in range(len(TP_count_list)):
                    file.write('{}, {}, {}, {}\n'.format(i, TP_count_list[i], FP_count_list[i], FN_count_list[i]))

        print('Best Current Epoch Selection:')
        print('Test Loss:')
        test_loss = test_loss_dict[loss_keys[0]]
        print(test_loss)
        print('Current Learning Rate = {}'.format(scheduler.get_last_lr()))
        min_test_loss = np.min(test_loss)
        error_from_best = np.abs(test_loss - min_test_loss)
        error_from_best[error_from_best < CONVERGENCE_TOLERANCE] = 0
        best_epoch = np.where(error_from_best == 0)[0][0]  # unpack numpy array, select first time since that value has happened
        print('Best epoch: {}'.format(best_epoch))
        print('Current Learning Rate = {}'.format(scheduler.get_last_lr()))
        print('LR_change_epoch = {}'.format(LR_change_epoch))
        print('CONVERGENCE_TOLERANCE = {}'.format(CONVERGENCE_TOLERANCE))
        print('nb_learning_rate_reductions = {}'.format(nb_learning_rate_reductions))

        # determine if to record a new checkpoint based on best test loss
        if (len(test_loss) - 1) == np.argmin(test_loss):
            torch.save({
                'epoch': epoch,
                'model_state_dict': yolo_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(torch_model_ofp, 'yolov3.ckpt'))

        # if delta between best epoch and current epoch > 10, stop
        if epoch - best_epoch >= early_stopping_count and (epoch - LR_change_epoch) >= early_stopping_count:
            # reduce the learning rate
            print('Loss Plateau at epoch {}, multiplying learning rate by 0.1'.format(epoch))
            scheduler.step()
            # modify convergence threshold to keep pace with the learning rate decay
            CONVERGENCE_TOLERANCE = CONVERGENCE_TOLERANCE * 0.1
            nb_learning_rate_reductions = nb_learning_rate_reductions + 1
            LR_change_epoch = epoch
            # only do this N times before stopping
            if nb_learning_rate_reductions > 2:
                # perform early stopping
                done_training = True
        #
        # if epoch - best_epoch > early_stopping_count:
        #     break  # break the epoch loop
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
    parser.add_argument('--use_amp', dest='use_amp', type=int, help='whether to use AMP [0 = false, 1 = true]', default=0)

    args = parser.parse_args()

    batch_size = args.batch_size
    train_database_filepath = args.train_database_filepath
    test_database_filepath = args.test_database_filepath
    output_folder = args.output_folder
    early_stopping = args.early_stopping
    learning_rate = args.learning_rate
    use_augmentation = args.use_augmentation
    use_amp = bool(args.use_amp)

    print('Arguments:')
    print('batch_size = {}'.format(batch_size))
    print('train_database_filepath = {}'.format(train_database_filepath))
    print('test_database_filepath = {}'.format(test_database_filepath))
    print('output folder = {}'.format(output_folder))
    print('early_stopping = {}'.format(early_stopping))
    print('learning_rate = {}'.format(learning_rate))
    print('use_augmentation = {}'.format(use_augmentation))
    print('use_amp = {}'.format(use_amp))

    config = dict()
    config['anchors'] = [[75, 75]]
    config['anchors_mask'] = [[0], [0], [0]]
    # config['anchors'] = [[64, 64], [128, 128]]
    # config['anchors'] = [(32, 32), (128, 128), (256, 256)]
    # config['anchors_mask'] = [[0, 1], [0, 1], [0, 1]]
    config['batch_size'] = batch_size
    config['learning_rate'] = learning_rate
    config['train_lmdb_filepath'] = train_database_filepath
    config['test_lmdb_filepath'] = test_database_filepath
    config['augment'] = use_augmentation
    config['generate_confusion'] = True

    train_model(config, output_folder, early_stopping, use_amp)
    print('Done training')
