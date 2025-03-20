import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler, optimizer
from torch import nn
import utils
from models import helper
from models.aggregators.gem_sem import GeMSem
from models.normalization import L2Norm

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.register_buffer()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.relu1 = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class BasicBlockSem(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(BasicBlockSem, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_planes)  #通道注意力
        self.sa = SpatialAttention() # 空间注意力模块

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        # Channel Attention Module
        out = self.ca(out) * out  #通道注意力
        out = self.sa(out) * out  # 空间注意力

        out = self.relu(out)

        return out
class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                 #---- sem
                 num_classes=19,
                 output_stride=16,
                 #---- Backbone
                 backbone_arch='resnet50',
                 backbone_config={},

                 #---- Aggregator
                 agg_arch='ConvAP',
                 agg_config={},

                 #---- Train hyperparameters
                 lr=0.03,
                 optimizer='sgd',
                 weight_decay=1e-3,
                 momentum=0.9,
                 lr_sched='linear',
                 lr_sched_args = {
                     'start_factor': 1,
                     'end_factor': 0.2,
                     'total_iters': 4000,
                 },

                 #----- Loss
                 loss_name='MultiSimilarityLoss',
                 miner_name='MultiSimilarityMiner',
                 miner_margin=0.1,
                 faiss_gpu=False
                 ):
        super().__init__()
        #---- sem
        self.num_classes = num_classes
        self.output_stride = output_stride

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config

        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        self.save_hyperparameters() # write hyperparams into a file

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level

        self.faiss_gpu = faiss_gpu

        # ----------------------------------
        # get the backbone and the aggregator
        self.semantic_extractor = helper.get_semantic_extractor(num_classes, output_stride)
        # self.depth = DepthAnything('vitb')
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

        # For validation in Lightning v2.0.0
        self.val_outputs = []
        self.agg = GeMSem()
        self.L2Norm = L2Norm(dim=1)
        # --------------------------------#
        #        Semantic Branch          #
        # ------------------------------- #
        self.in_block_sem = nn.Sequential(
            nn.Conv2d(19, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.in_block_sem_1 = BasicBlockSem(64, 128, kernel_size=3, stride=2, padding=1)
        self.in_block_sem_2 = BasicBlockSem(128, 256, kernel_size=3, stride=2, padding=1)
        self.in_block_sem_3 = BasicBlockSem(256, 512, kernel_size=3, stride=2, padding=1)

        # --------------------------------#
        #         Attention Module        #
        # ------------------------------- #
        # Final Scene Classification Layers
        self.lastConvRGB1 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=5, stride=2,padding=0,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.lastConvRGB2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.lastConvSEM1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.lastConvSEM2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.L2Norm = L2Norm(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool3 = nn.AvgPool2d(3, stride=1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, 2048)


    # the forward pass of the lightning model
    def forward(self, x):
        sem = self.semantic_extractor(x)
        x = self.backbone(x)
        e4, g_fea = x
        y = self.in_block_sem(sem)
        y1 = self.in_block_sem_1(y)
        y2 = self.in_block_sem_2(y1)
        y3 = self.in_block_sem_3(y2)
        # --------------------------------#
        #         Attention Module        #
        # ------------------------------- #
        e5 = self.lastConvRGB1(e4)
        e6 = self.lastConvRGB2(e5)

        y4 = self.lastConvSEM1(y3)
        y5 = self.lastConvSEM2(y4)

        # Attention Mechanism
        e7 = e6 * self.sigmoid(y5)

        # --------------------------------#
        #            Classifier           #
        # ------------------------------- #
        e8 = self.avgpool3(e7)
        act = e8.view(e8.size(0), -1)
        act = self.dropout(act)
        act = self.fc(act)
        final = torch.cat((g_fea, act), dim=1)
        final = self.L2Norm(final)

        return final


    # configure the optimizer
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')


        if self.lr_sched.lower() == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sched_args['milestones'], gamma=self.lr_sched_args['gamma'])
        elif self.lr_sched.lower() == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args['T_max'])
        elif self.lr_sched.lower() == 'linear':
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_sched_args['start_factor'],
                end_factor=self.lr_sched_args['end_factor'],
                total_iters=self.lr_sched_args['total_iters']
            )

        return [optimizer], [scheduler]

    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self,  epoch, batch_idx, optimizer, optimizer_closure):
        # warm up lr
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)

            # calculate the % of trivial pairs/triplets
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple:
                # somes losses do the online mining inside (they don't need a miner objet),
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class,
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                 len(self.batch_acc), prog_bar=True, logger=True)
        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch

        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape

        # reshape places and labels
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images) # Here we are calling the method forward that we defined above

        if torch.isnan(descriptors).any():
            raise ValueError('NaNs in descriptors')

        loss = self.loss_function(descriptors, labels) # Call the loss_function we defined above

        self.log('loss', loss.item(), logger=True, prog_bar=True)
        return {'loss': loss}

    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        places, _ = batch
        descriptors = self(places)
        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()

    def on_validation_epoch_start(self):
        # reset the outputs list
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]

    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.val_outputs

        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]

        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)

            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.ground_truth
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif 'sped' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.ground_truth
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            pitts_dict = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu
            )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')

        # reset the outputs list
        self.val_outputs = []