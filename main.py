import pytorch_lightning as pl
from vpr_model import VPRModel
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule

if __name__ == '__main__':        
    datamodule = GSVCitiesDataModule(
        batch_size=60,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False, # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=(224, 224),
        num_workers=10,
        show_data_stats=True,
        val_set_names=['pitts30k_test','msls_val'],
        # val_set_names=['msls_val']    # pitts30k_val, pitts30k_test, msls_val
    )
    
    model = VPRModel(
        #---- Encoder
        num_classes=19,
        output_stride=16,
        backbone_arch='dinov2_vitl14',
        backbone_config={
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        },
        agg_arch='addse',
        agg_config={
            'num_channels': 1024,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        },
        lr=6e-5,
        optimizer='adamw',
        weight_decay=9.5e-9, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.1,
            'total_iters': 10000,
        },

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='msls_val/R1',
        filename=f'{model.encoder_arch}' + '_({epoch:02d})_R1[{pitts30k_test/R1:.4f}]_R1[{msls_val/R1:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        save_last=False,
        mode='max'
    )


    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        default_root_dir=f'./logs/', # Tensorflow can be used to viz
        num_nodes=1,
        num_sanity_val_steps=0, # runs a validation step before stating training
        precision='16-mixed', # we use half precision to reduce  memory usage
        max_epochs=20,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20
    )

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)