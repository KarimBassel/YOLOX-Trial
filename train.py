import os
import argparse
import torch
from loguru import logger
from yolox.exp import get_exp
from yolox.core import Trainer, launch
from yolox.utils import configure_nccl, configure_omp, setup_logger, get_num_devices
import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch
from yolox.exp import Exp, check_exp_value, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices
class CustomTrainer(Trainer):
    def __init__(self, exp, args, train_loader):
        print("Initializing CustomTrainer...")  # Debugging point
        print("Experiment Configuration:")
        print(f"Data Directory: {exp.data_dir}")
        print(f"Training Annotation: {exp.train_ann}")
        print(f"Validation Annotation: {exp.val_ann}")
        print(f"Number of Classes: {exp.num_classes}")
        print(f"Input Size: {exp.input_size}")
        
        try:
            super().__init__(exp, args)  # Call the parent class's init method and pass args
            print("CustomTrainer initialized.")  # Debugging print
            
            # Ensure model is initialized after calling super().__init__()
            if not hasattr(self, 'model'):
                print("Model attribute is missing in the parent class!")
                # Manually initialize the model if it's not set
                self.model = exp.get_model()
                print("Model manually initialized.")
                
            if not hasattr(self.model, 'forward'):
                raise AttributeError("The 'model' does not have a 'forward' method. Initialization failed.")
            
            # Move model to GPU (if using CUDA)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)  # Move model to GPU or CPU
            print(f"Model moved to {self.device}.")

            # Set the train_loader
            self.train_loader = train_loader
            print("CustomTrainer initialized with train_loader.")  # Debugging print
            
            self.optimizer = exp.get_optimizer(self.model)

        except Exception as e:
            print(f"Error during initialization of CustomTrainer: {e}")  # Print the error message
            raise  # Reraise the exception if you want to stop execution after the error
        print("CustomTrainer initialized!")  # Debugging point

    
    def train_one_epoch(self, batch_size, fp16):
        """
        Implements the training loop for a single epoch.
        """
        self.model.train()  # Set the model to training mode
        total_loss = 0.0

        # Debugging: Start of training loop
        print("Starting training for one epoch...")  # Debugging point
        
        for iteration, batch in enumerate(self.train_loader):
            # Unpack the batch manually
            images = batch[0]  # [batch_size, 3, height, width]
            targets = batch[1]  # [batch_size, max_objects, 5]
            extra_data = batch[2]  # Additional info, could be image IDs or metadata
            additional_info = batch[3]  # Possible auxiliary info (e.g., target for additional loss)

            print(f"Batch Structure: {type(batch)}, Length: {len(batch)}")
            print(f"Images shape: {images.shape}")
            print(f"Targets shape: {targets.shape}")
            print(f"Extra Data: {extra_data}")  # Check the contents of extra data
            print(f"Additional Info: {additional_info.shape}")


            # Proceed with the usual training procedure
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            print(f"Outputs shape: {outputs.shape}")  # Debugging point

            # Compute the loss (depending on your model setup, this might be adjusted)
            try:
                loss = self.model.loss(outputs, targets)
                print(f"Loss computed successfully: {loss.item()}")  # Debugging point
            except Exception as e:
                print(f"Error during loss computation: {e}")
                continue  # Skip this batch if there is an error during loss computation

            # Backward pass
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Accumulate loss for logging
            total_loss += loss.item()

            if iteration % self.exp.print_interval == 0:
                print(f"Iteration {iteration}/{len(self.train_loader)} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss



    def log_epoch_details(self, epoch, total_epochs, train_loss, val_loss, mAP):
        """
        Logs the details of the current epoch after each training step.
        """
        logger.info(f"Epoch {epoch}/{total_epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Validation Loss: {val_loss:.4f}, "
                    f"mAP: {mAP:.4f}")
        print(f"Epoch {epoch}/{total_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}, "
              f"mAP: {mAP:.4f}")  # Debugging point

    def train(self, batch_size, fp16, occupy):
        try:
            total_epochs = self.exp.max_epoch
            print("Starting training loop...")  # Debugging point
            
            for epoch in range(total_epochs):
                print(f"Epoch {epoch+1}/{total_epochs} starting...")  # Debugging point
                
                # Train one epoch
                try:
                    train_loss = self.train_one_epoch(batch_size, fp16)
                    print(f"Train loss for epoch {epoch+1}: {train_loss}")  # Debugging point
                except Exception as e:
                    print(f"Error during training of epoch {epoch+1}: {e}")
                    continue  # Skip to next epoch if error occurs

                # Validate and calculate metrics
                try:
                    val_loss, mAP = self.validate()
                    print(f"Validation loss: {val_loss}, mAP: {mAP}")  # Debugging point
                except Exception as e:
                    print(f"Error during validation of epoch {epoch+1}: {e}")
                    val_loss, mAP = None, None  # Placeholder in case of validation error

                # Log epoch details
                self.log_epoch_details(epoch + 1, total_epochs, train_loss, val_loss, mAP)

                # Save checkpoint at intervals
                if (epoch + 1) % self.exp.eval_interval == 0:
                    self.save_checkpoint(epoch)
                    print(f"Checkpoint saved for epoch {epoch+1}")  # Debugging point

        except Exception as e:
            print(f"Error during training: {e}")  # Print the error message

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help="Caching imgs to ram/disk for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics. \
                Implemented loggers include `tensorboard`, `mlflow` and `wandb`.",
        default="tensorboard"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

@logger.catch
def main(exp: Exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = exp.get_trainer(args)
    trainer.train()


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    check_exp_value(exp)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )