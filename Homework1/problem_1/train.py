import wandb
import os
from utils import set_seed
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import argparse
from torchvision.models import resnet18
from tqdm import tqdm


def get_scheduler(use_scheduler, optimizer, **kwargs):
    """
    :param use_scheduler: whether to use lr scheduler
    :param optimizer: instance of optimizer
    :param kwargs: other args to pass to scheduler; already filled with some default values in train_model()
    :return: scheduler
    """
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
    if use_scheduler:
        raise NotImplementedError
    else:
        scheduler = None
    return scheduler


def evaluate(model, data_loader, device):
    """
    :param model: instance of model
    :param data_loader: instance of data loader
    :param device: cpu or cuda
    :return: accuracy, cross entropy loss (sum)
    """
    # code below is just a reference, you may modify this part during your implementation
    model.eval()
    num_instances = 0
    val_acc = None
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        raise NotImplementedError
    model.train()
    return val_acc, val_loss


def train_model(
        run_name,
        model,
        batch_size,
        epochs,
        learning_rate,
        device,
        save_dir,
        use_scheduler,
):
    model.to(device)

    # Complete the code below to load the dataset; you can customize the dataset class or use ImageFolder
    # Note that in your transform, you should include resize the image to 224x224, and normalize the image with appropriate mean and std
    train_set = None
    val_set = None
    test_set = None


    n_train, n_val, n_test = len(train_set), len(val_set), len(test_set)
    loader_args = dict(batch_size=batch_size, num_workers=4)
    batch_steps = n_train // batch_size
    total_training_steps = epochs * batch_steps

    train_loader = DataLoader(train_set, shuffle=True, **loader_args, drop_last=True)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    # Initialize a new wandb run and log experiment config parameters; don't forget the run name
    # you can also set run name to reflect key hyperparameters, such as learning rate, batch size, etc.: run_name = f'lr_{learning_rate}_bs_{batch_size}...'
    # code here


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(use_scheduler, optimizer, max_lr=learning_rate,
                              total_steps=total_training_steps, pct_start=0.1, final_div_factor=10)

    criterion = nn.CrossEntropyLoss()

    # record necessary metrics
    global_step = 0
    seen_examples = 0
    best_val_loss = float('inf')

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        with tqdm(total=batch_steps * batch_size, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for inputs, labels in train_loader:
                seen_examples += inputs.size(0)
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if use_scheduler:
                    scheduler.step()
                pbar.update(inputs.shape[0])
                global_step += 1
                # save necessary metrics in a dictionary; it's recommended to also log seen_examples, which helps you creat appropriate figures in Part 3
                # code here

                if global_step % batch_steps == 0:
                    # evaluate on validation set
                    val_acc, val_loss = evaluate(model, val_loader, device)
                    # update metrics from validation results in the dictionary
                    # code here

                    if best_val_loss > val_loss:
                        best_val_loss = val_loss
                        os.makedirs(os.path.join(save_dir, f'{run_name}_{rid}'), exist_ok=True)
                        state_dict = model.state_dict()
                        torch.save(state_dict, os.path.join(save_dir, f'{run_name}_{rid}', 'checkpoint.pth'))
                        print(f'Checkpoint at step {global_step} saved!')
                # log metrics to wandb
                # code here

                pbar.set_postfix(**{'loss (batch)': loss.item()})

    # load best checkpoint and evaluate on test set
    print(f'training finished, run testing using best ckpt...')
    state_dict = torch.load(os.path.join(save_dir, f'{run_name}_{rid}', 'checkpoint.pth'))
    model.load_state_dict(state_dict)
    test_acc, test_loss = evaluate(model, test_loader, device)

    # log test results to wandb
    # code here


def get_args():
    parser = argparse.ArgumentParser(description='E2EDL training script')
    # exp description
    parser.add_argument('--run_name', type=str, default='baseline',
                        help="a brief description of the experiment; "
                             "alternatively, you can set the name automatically based on hyperparameters:"
                             "run_name = f'lr_{learning_rate}_bs_{batch_size}...' to reflect key hyperparameters")
    # dirs
    parser.add_argument('--save_dir', type=str, default='./checkpoints/',
                        help='save best checkpoint to this dir')
    # training config
    parser.add_argument('--epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size; modify this to fit your GPU memory')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use_scheduler', action='store_true', help='use lr scheduler')

    # IMPORTANT: if you are copying this script to notebook, replace 'return parser.parse_args()' with 'args = parser.parse_args("")'

    return parser.parse_args()



if __name__ == '__main__':
    rid = random.randint(0, 1000000)
    set_seed(42)
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(pretrained=False, num_classes=37)
    train_model(
        run_name=args.run_name,
        model=model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        save_dir=args.save_dir,
        use_scheduler=args.use_scheduler,
    )
