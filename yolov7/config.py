_base_ = './yolov7_l_syncbn_fast_8x16b-300e_coco.py'

max_epochs = 100  # 训练的最大 epoch
data_root = 'data/dataset_single_folder'  # 数据集目录的绝对路径

# 结果保存的路径，可以省略，省略保存的文件名位于 work_dirs 下 config 同名的文件夹中
# 如果某个 config 只是修改了部分参数，修改这个变量就可以将新的训练文件保存到其他地方
work_dir = './uncropped'

# load_from 可以指定本地路径或者 URL，设置了 URL 会自动进行下载，因为上面已经下载过，我们这里设置本地路径
load_from = 'runs/train/uncropped/weights/best.pt'

train_batch_size_per_gpu = 8  # 根据自己的GPU情况，修改 batch size，YOLOv5-s 默认为 8卡 * 16bs
train_num_workers = 8  # 推荐使用 train_num_workers = nGPU x 4

save_epoch_intervals = 10  # 每 interval 轮迭代进行一次保存一次权重

# 根据自己的 GPU 情况，修改 base_lr，修改的比例是 base_lr_default * (your_bs / default_bs)
base_lr = 0.1

num_classes = 6
metainfo = dict(  # 根据 class_with_id.txt 类别信息，设置 metainfo
    CLASSES=('scallop','herring','dead-scallop','flounder','roundfish','skate'),
    PALETTE=[(220, 20, 60)]  # 画图时候的颜色，随便设置即可
)

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=10,  # 第几个epoch后验证，这里设置 10 是因为前 10 个 epoch 精度不高，测试意义不大，故跳过
    val_interval=save_epoch_intervals  # 每 val_interval 轮迭代进行一次测试评估
)

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),

        # loss_cls 会根据 num_classes 动态调整，但是 num_classes = 1 的时候，loss_cls 恒为 0
        loss_cls=dict(loss_weight=0.5 * (num_classes / 6 * 3 / _base_.num_det_layers))
    )
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,  # 数据量太少的话，可以使用 RepeatDataset 来增量数据，这里设置 5 是 5 倍
        dataset=dict(
            type=_base_.dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='annotations/train.json',
            data_prefix=dict(img='images/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=_base_.train_pipeline)
    ))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator

optim_wrapper = _base_.optim_wrapper

default_hooks = dict(
    # 设置间隔多少个 epoch 保存模型，以及保存模型最多几个，`save_best` 是另外保存最佳模型（推荐）
    checkpoint=dict(type='CheckpointHook', interval=save_epoch_intervals,
                    max_keep_ckpts=5, save_best='auto'),
    # logger 输出的间隔
    logger=dict(type='LoggerHook', interval=10)
)