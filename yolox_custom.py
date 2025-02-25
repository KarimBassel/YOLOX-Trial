import os
import torch
import torch.nn as nn
from yolox.exp import Exp as MyExp
from torch.optim import Adam, SGD
from yolox.data import COCODataset, TrainTransform, DataLoader, InfiniteSampler, YoloBatchSampler


class Exp(MyExp):
    def __init__(self):
        print("Initializing custom Experiment...")  # Debugging point
        super(Exp, self).__init__()

        # ---------------- Dataset Settings ---------------- #
        self.data_dir = "D:\YOLOX Dataset"  # Dataset root
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.train_img_dir = "images/train"
        self.val_img_dir = "images/val"

        # self.data_dir = r"C:\Users\Karim Bassel\Downloads\coco128\coco128"  # Dataset root
        # self.train_ann = "instances_train2017.json"
        # self.val_ann = "instances_val2017.json"
        # self.train_img_dir = "train2017"
        # self.val_img_dir = "val2017"
        
        print(f"Dataset settings: data_dir={self.data_dir}, train_ann={self.train_ann}, val_ann={self.val_ann}")  # Debugging point

        # ---------------- Model Settings ---------------- #
        self.num_classes = 43  # Change this to match your dataset classes
        #self.num_classes = 71  # Change this to match your dataset classes
        self.depth = 0.33  # Model depth (YOLOX-S)
        self.width = 0.50  # Model width (YOLOX-S)
        
        print(f"Model settings: num_classes={self.num_classes}, depth={self.depth}, width={self.width}")  # Debugging point

        # ---------------- Training Settings ---------------- #
        self.max_epoch = 100  # Total training epochs
        self.data_num_workers = 4  # Number of CPU workers
        self.eval_interval = 5  # Run validation every 5 epochs
        self.warmup_epochs = 5  # Warmup phase
        
        print(f"Training settings: max_epoch={self.max_epoch}, data_num_workers={self.data_num_workers}, "
              f"eval_interval={self.eval_interval}, warmup_epochs={self.warmup_epochs}")  # Debugging point

        # ---------------- Optimizer & Learning Rate ---------------- #
        self.basic_lr_per_img = 0.01 / 64.0
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.optimizer_type = "adam"
        
        print(f"Optimizer settings: basic_lr_per_img={self.basic_lr_per_img}, momentum={self.momentum}, "
              f"weight_decay={self.weight_decay}")  # Debugging point

        # ---------------- Advanced Settings ---------------- #
        self.no_aug_epochs = 10  # Last 10 epochs without augmentation
        self.ema = True  # Enable EMA (Exponential Moving Average)
        self.input_size = (640, 640)
        self.test_size = (640, 640)

        print(f"Advanced settings: no_aug_epochs={self.no_aug_epochs}, ema={self.ema}, "
              f"input_size={self.input_size}, test_size={self.test_size}")  # Debugging point

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    # def get_data_loader(self, batch_size, is_distributed):
    #     print(f"Creating data loader with batch size={batch_size}, distributed={is_distributed}")  # Debugging point


    #     try:
    #         print(f"Creating data loader with batch_size={batch_size}, is_distributed={is_distributed}")  # Debugging point
    #         self.dataset = COCODataset(
    #             data_dir=self.data_dir,
    #             json_file=self.train_ann,
    #             img_size=self.input_size,
    #             preproc=TrainTransform(max_labels=43, flip_prob=0.5, hsv_prob=1.0)
    #         )
    #         print(f"Dataset loaded: {len(self.dataset)} samples.")  # Debugging point
    #     except Exception as e:
    #         print(f"Error loading dataset: {e}")  # Print any errors during dataset loading
    #         return None
    #     sampler = InfiniteSampler(len(self.dataset), seed=0)
    #     print(f"Sampler created with {len(self.dataset)} samples.")  # Debugging point
        
    #     batch_sampler = YoloBatchSampler(
    #         sampler=sampler,
    #         batch_size=batch_size,
    #         drop_last=False
    #     )

    #     dataloader = DataLoader(
    #         self.dataset,
    #         batch_sampler=batch_sampler,
    #         num_workers=self.data_num_workers
    #     )

    #     print("Data loader created successfully.")  # Debugging point

    #     return dataloader

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        """
        Get dataloader according to cache_img parameter.
        Args:
            no_aug (bool, optional): Whether to turn off mosaic data enhancement. Defaults to False.
            cache_img (str, optional): cache_img is equivalent to cache_type. Defaults to None.
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
                None: Do not use cache, in this case cache_data is also None.
        """
        from yolox.data import (
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        # if cache is True, we will create self.dataset before launch
        # else we will create self.dataset after launch
        if self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None, \
                    "cache_img must be None if you didn't create self.dataset before launch"
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)

        self.dataset = MosaicDetection(
            dataset=self.dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader
    def get_train_data_loader(self, batch_size, is_distributed):
        """
        This method should explicitly call get_data_loader and return the train loader.
        """
        print("Getting train data loader...")
        return self.get_data_loader(batch_size, is_distributed)
