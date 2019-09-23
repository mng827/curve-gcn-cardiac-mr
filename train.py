import torch
import torch.optim as optim

from tqdm import tqdm
import os
import numpy as np

from misc import plot_utils
import losses
import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model, train_loader, test_loader, config, logger):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.logger = logger

        self.model.to(device)
        self.global_step = 0

        self.fixed_image, self.fixed_contour, self.fixed_contour_binary = iter(self.test_loader).next()
        self.fixed_image = np.transpose(self.fixed_image, axes=(0, 3, 1, 2))

        self.fixed_image = torch.Tensor(self.fixed_image).to(device)
        self.fixed_contour = torch.Tensor(self.fixed_contour).to(device)
        self.fixed_contour_binary = torch.Tensor(self.fixed_contour_binary).to(device)
        self.fixed_contour = self.fixed_contour / self.config.image_size

        self.initial_polygon1 = torch.stack([0.1 * torch.cos(torch.linspace(0, 2 * np.pi, 101)[:-1]),
                                             0.1 * torch.sin(torch.linspace(0, 2 * np.pi, 101)[:-1])], dim=-1)
        self.initial_polygon1 = torch.unsqueeze(self.initial_polygon1, dim=0) + 0.5

        self.initial_polygon2 = torch.stack([0.2 * torch.cos(torch.linspace(0, 2 * np.pi, 101)[:-1]),
                                             0.2 * torch.sin(torch.linspace(0, 2 * np.pi, 101)[:-1])], dim=-1)
        self.initial_polygon2 = torch.unsqueeze(self.initial_polygon2, dim=0) + 0.5

        self.initial_polygon3 = torch.stack([0.2 * torch.cos(torch.linspace(0, 2 * np.pi, 101)[:-1]),
                                             0.2 * torch.sin(torch.linspace(0, 2 * np.pi, 101)[:-1])], dim=-1)
        self.initial_polygon3 = torch.unsqueeze(self.initial_polygon3, dim=0) + torch.Tensor([0.35, 0.5])

        self.optimizer = optim.Adam(self.model.parameters(), config.learning_rate, [config.beta1, config.beta2])

        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.decay_every_itr, gamma=0.1)


    def train(self):
        for cur_epoch in range(self.config.epoch):
            self.logger.info('epoch: {}'.format(int(cur_epoch)))

            self.train_epoch()
            if (cur_epoch + 1) % 5 == 0:
                self.test_epoch(cur_epoch, plot=True)

            if (cur_epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(cur_epoch + 1)


    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_decay': self.lr_decay.state_dict()
        }

        save_name = os.path.join(self.config.checkpoint_dir, 'model.pth')
        torch.save(save_state, save_name)
        print('Saved model: {}'.format(save_name))


    def resume(self, path):
        save_state = torch.load(path)
        self.global_step = save_state['global_step']
        self.epoch = save_state['epoch']
        self.model.load_state_dict(save_state['state_dict'])
        self.optimizer.load_state_dict(save_state['optimizer'])
        self.lr_decay.load_state_dict(save_state['lr_decay'])
        print('Loaded model: {}'.format(path))


    def train_epoch(self):
        loss_list = []
        bce_loss_list = []
        dice_list = []
        accuracy_list = []

        self.model.train()

        for itr, (x, gt, gt_binary) in enumerate(tqdm(self.train_loader)):
            x = np.transpose(x, axes=(0,3,1,2))
            gt = gt / self.config.image_size
            x, gt, gt_binary = torch.Tensor(x).to(device), torch.Tensor(gt).to(device), torch.Tensor(gt_binary).to(device)

            pred = self.model(x,
                              self.initial_polygon1.repeat(x.size(0), 1, 1),
                              self.initial_polygon2.repeat(x.size(0), 1, 1),
                              self.initial_polygon3.repeat(x.size(0), 1, 1),
                              num_polys=3)

            pred_binary = (pred['detection_logits'] > 0)

            _, loss1 = losses.polygon_matching_loss(100, pred['pred_polys1'][-1], gt[:,:,:,0], "L2", gt_binary[:,0])
            _, loss2 = losses.polygon_matching_loss(100, pred['pred_polys2'][-1], gt[:,:,:,1], "L2", gt_binary[:,1])
            _, loss3 = losses.polygon_matching_loss(100, pred['pred_polys3'][-1], gt[:,:,:,2], "L2", gt_binary[:,2])

            bce_loss1 = losses.binary_classification_loss(pred['detection_logits'][:,0], gt_binary[:,0])
            bce_loss2 = losses.binary_classification_loss(pred['detection_logits'][:,1], gt_binary[:,1])
            bce_loss3 = losses.binary_classification_loss(pred['detection_logits'][:,2], gt_binary[:,2])

            total_loss = loss1 + loss2 + loss3 + bce_loss1 + bce_loss2 + bce_loss3

            total_loss.backward()
            self.global_step += 1
            self.optimizer.step()
            self.optimizer.zero_grad()

            dice = np.zeros([3])
            accuracy = np.zeros([3])

            for p in range(3):
                pred_mask = metrics.mask_from_poly(pred['pred_polys{}'.format(p+1)][-1].cpu().data.numpy() * self.config.image_size,
                                                   self.config.image_size, self.config.image_size,
                                                   pred_binary[:,p].cpu().data.numpy().astype(np.uint8))

                gt_mask = metrics.mask_from_poly(gt[:,:,:,p].cpu().data.numpy() * self.config.image_size,
                                                 self.config.image_size, self.config.image_size,
                                                 gt_binary[:,p].cpu().data.numpy().astype(np.uint8))

                dice[p] = metrics.dice_3d_from_mask(pred_mask, gt_mask)
                accuracy[p] = metrics.detection_accuracy(pred_binary[:,p].cpu().data.numpy().astype(np.uint8),
                                                         gt_binary[:,p].cpu().data.numpy().astype(np.uint8))

            loss_list.append([loss1.data, loss2.data, loss3.data])
            bce_loss_list.append([bce_loss1.data, bce_loss2.data, bce_loss3.data])
            dice_list.append(dice)
            accuracy_list.append(accuracy)

        avg_loss = np.mean(loss_list, axis=0)
        avg_bce_loss = np.mean(bce_loss_list, axis=0)
        avg_dice = np.mean(dice_list, axis=0)
        avg_accuracy = np.mean(accuracy_list, axis=0)

        self.logger.info("train | loss: {} | bce_loss: {} | dice: {} | detection_accuracy: {}".
                         format(avg_loss, avg_bce_loss, avg_dice, avg_accuracy))


    def test_epoch(self, cur_epoch, plot):
        loss_list = []
        bce_loss_list = []
        dice_list = []
        accuracy_list = []

        self.model.eval()

        for itr, (x, gt, gt_binary) in enumerate(tqdm(self.test_loader)):
            x = np.transpose(x, axes=(0,3,1,2))               # NCHW
            gt = gt / self.config.image_size
            x, gt, gt_binary = torch.Tensor(x).cuda(), torch.Tensor(gt).cuda(), torch.Tensor(gt_binary).cuda()

            pred = self.model(x,
                              self.initial_polygon1.repeat(x.size(0), 1, 1),
                              self.initial_polygon2.repeat(x.size(0), 1, 1),
                              self.initial_polygon3.repeat(x.size(0), 1, 1),
                              num_polys=3)

            pred_binary = (pred['detection_logits'] > 0)

            _, loss1 = losses.polygon_matching_loss(100, pred['pred_polys1'][-1], gt[:, :, :, 0], "L2", gt_binary[:, 0])
            _, loss2 = losses.polygon_matching_loss(100, pred['pred_polys2'][-1], gt[:, :, :, 1], "L2", gt_binary[:, 1])
            _, loss3 = losses.polygon_matching_loss(100, pred['pred_polys3'][-1], gt[:, :, :, 2], "L2", gt_binary[:, 2])

            bce_loss1 = losses.binary_classification_loss(pred['detection_logits'][:, 0], gt_binary[:, 0])
            bce_loss2 = losses.binary_classification_loss(pred['detection_logits'][:, 1], gt_binary[:, 1])
            bce_loss3 = losses.binary_classification_loss(pred['detection_logits'][:, 2], gt_binary[:, 2])

            dice = np.zeros([3])
            accuracy = np.zeros([3])

            for p in range(3):
                pred_mask = metrics.mask_from_poly(pred['pred_polys{}'.format(p+1)][-1].cpu().data.numpy() * self.config.image_size,
                                                   self.config.image_size, self.config.image_size,
                                                   pred_binary[:,p].cpu().data.numpy().astype(np.uint8))

                gt_mask = metrics.mask_from_poly(gt[:,:,:,p].cpu().data.numpy() * self.config.image_size,
                                                 self.config.image_size, self.config.image_size,
                                                 gt_binary[:,p].cpu().data.numpy().astype(np.uint8))

                dice[p] = metrics.dice_3d_from_mask(pred_mask, gt_mask)
                accuracy[p] = metrics.detection_accuracy(pred_binary[:,p].cpu().data.numpy().astype(np.uint8),
                                                         gt_binary[:,p].cpu().data.numpy().astype(np.uint8))

            # pred_combined = np.stack([torch.stack(pred['pred_polys1'], dim=-1).cpu().data.numpy(),
            #                           torch.stack(pred['pred_polys2'], dim=-1).cpu().data.numpy(),
            #                           torch.stack(pred['pred_polys3'], dim=-1).cpu().data.numpy()], axis=-1)
            #
            # plot_utils.plot_images_and_contour_evolution(x.cpu().numpy(),
            #                                              pred_combined * self.config.image_size,
            #                                              gt.cpu().data.numpy() * self.config.image_size,
            #                                              pred_binary.cpu().data.numpy(),
            #                                              gt_binary.cpu().data.numpy(),
            #                                              self.config.checkpoint_dir,
            #                                              'test-image-batch-{:03d}.svg'.format(itr))

            # pred_combined = np.stack([pred['pred_polys1'][-1].cpu().data.numpy(),
            #                           pred['pred_polys2'][-1].cpu().data.numpy(),
            #                           pred['pred_polys3'][-1].cpu().data.numpy()], axis=-1)
            #
            # plot_utils.plot_merged_images(x.cpu().numpy(),
            #                               pred_combined * self.config.image_size,
            #                               gt.cpu().data.numpy() * self.config.image_size,
            #                               pred_binary.cpu().data.numpy(),
            #                               gt_binary.cpu().data.numpy(),
            #                               self.config.checkpoint_dir,
            #                               'test-image-batch-{:03d}.svg'.format(itr))

            loss_list.append([loss1.data, loss2.data, loss3.data])
            bce_loss_list.append([bce_loss1.data, bce_loss2.data, bce_loss3.data])
            dice_list.append(dice)
            accuracy_list.append(accuracy)

        avg_loss = np.mean(loss_list, axis=0)
        avg_bce_loss = np.mean(bce_loss_list, axis=0)
        avg_dice = np.mean(dice_list, axis=0)
        avg_accuracy = np.mean(accuracy_list, axis=0)

        self.logger.info("test | loss: {} | bce_loss: {} | dice: {} | detection_accuracy: {}".
                         format(avg_loss, avg_bce_loss, avg_dice, avg_accuracy))

        if plot:
            pred = self.model(self.fixed_image,
                              self.initial_polygon1.repeat(self.fixed_image.size(0), 1, 1),
                              self.initial_polygon2.repeat(self.fixed_image.size(0), 1, 1),
                              self.initial_polygon3.repeat(self.fixed_image.size(0), 1, 1),
                              num_polys=3)

            pred_binary = (pred['detection_logits'] > 0)

            pred_combined = np.stack([pred['pred_polys1'][-1].cpu().data.numpy(),
                                      pred['pred_polys2'][-1].cpu().data.numpy(),
                                      pred['pred_polys3'][-1].cpu().data.numpy()], axis=-1)

            plot_utils.plot_merged_images(self.fixed_image.cpu().numpy(),
                                          pred_combined * self.config.image_size,
                                          self.fixed_contour.cpu().data.numpy() * self.config.image_size,
                                          pred_binary.cpu().data.numpy(),
                                          self.fixed_contour_binary.cpu().data.numpy(),
                                          self.config.checkpoint_dir,
                                          'test-image-epoch-{:03d}.svg'.format(cur_epoch))
