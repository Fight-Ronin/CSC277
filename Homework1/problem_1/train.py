import wandb
import os
from utils import set_seed
import random
import torch
from torch import nn, optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import argparse
from torchvision.models import resnet18
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd

def get_scheduler(use_scheduler, optimizer, **kwargs):
    """
    :param use_scheduler: whether to use lr scheduler
    :param optimizer: instance of optimizer
    :param kwargs: other args to pass to scheduler; already filled with some default values in train_model()
    :return: scheduler
    """
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
    if use_scheduler:
        # Properly implement the OneCycleLR scheduler
        # Assuming `max_lr`, `steps_per_epoch`, and `epochs` are passed in as keyword arguments
        max_lr = kwargs.get('max_lr', 0.01)  # Default max_lr if not specified
        total_steps_1 = kwargs.get('total_steps')  # Total steps must be provided or calculated
        if not total_steps_1:
            steps_per_epoch = kwargs.get('steps_per_epoch', 100)  # Default or passed number of steps per epoch
            epochs = kwargs.get('epochs', 10)  # Default or passed number of training epochs
            total_steps_1 = steps_per_epoch * epochs
        anneal_strategy = kwargs.get('anneal_strategy', 'cos')  # Default annealing strategy
        final_div_factor = kwargs.get('final_div_factor', 1e4)  # How much to reduce the lr at the end

        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps_1,
            # epochs=epochs,
            pct_start=0.3,  # Percentage of the cycle spent increasing the learning rate
            anneal_strategy=anneal_strategy,
            final_div_factor=final_div_factor
        )
    else:
        scheduler = None

    return scheduler

def scale_learning_rate_and_hyperparameters(learning_rate, batch_size, original_batch_size, beta1, beta2, epsilon):
    """
    Scale the learning rate and other hyperparameters according to the scaling rule for Adam.
    :param learning_rate: Original learning rate
    :param batch_size: New batch size
    :param original_batch_size: Original batch size
    :param beta1: Original beta1 parameter for Adam
    :param beta2: Original beta2 parameter for Adam
    :param epsilon: Original epsilon parameter for Adam
    :return: Scaled learning rate, beta1, beta2, epsilon
    """
    # Calculate the scaling factor
    kappa = batch_size / original_batch_size
    sqrt_kappa = kappa ** 0.5

    # Scale the learning rate
    scaled_lr = learning_rate * sqrt_kappa

    # Scale the beta parameters and epsilon
    scaled_beta1 = 1 - kappa * (1 - beta1)
    scaled_beta2 = 1 - kappa * (1 - beta2)
    scaled_epsilon = epsilon / sqrt_kappa

    return scaled_lr, scaled_beta1, scaled_beta2, scaled_epsilon

def evaluate(model, data_loader, device):
    """
    :param model: instance of model  
    :param data_loader: instance of data loader
    :param device: cpu or cuda
    :return: accuracy, cross entropy loss (sum)
    """
    # Modified eval. function
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():  # No need to track gradients for evaluation
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    model.train()  # Set the model back to training mode
    return accuracy, total_loss, loss.item()

 # Define custom Dataset class
class OxfordPetsDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        # Creating a mapping of labels to integers
        self.label_map = {label: idx for idx, label in enumerate(dataframe['label'].unique())}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        # Use the map to convert label strings to integers
        label = self.label_map[self.dataframe.iloc[idx]['label']]
        image = Image.open(img_path).convert('RGB')  # Convert image to RGB

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model(
        run_name,
        model,
        batch_size,
        epochs,
        learning_rate,
        device,
        save_dir,
        use_scheduler,

        original_batch_size = None,
        scale_lr=False,
        beta1=0.9,  # Default values for Adam
        beta2=0.999,
        epsilon=1e-8
):
    model.to(device)

    # Adjust learning rate and other parameters if needed
    if scale_lr and batch_size is not None:
        learning_rate, beta1, beta2, epsilon = scale_learning_rate_and_hyperparameters(
            learning_rate, batch_size, original_batch_size, beta1, beta2, epsilon
        )

    # Complete the code below to load the dataset; you can customize the dataset class or use ImageFolder
    # Note that in your transform, you should include resize the image to 224x224, and normalize the image with appropriate mean and std

    df = pd.read_csv('problem_1/oxford_pet_split.csv')
    df['image_path'] = df['image_name'].apply(lambda x: os.path.join('data/images', x))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 as required by ResNet-18
        transforms.ToTensor(),  # Convert images to PyTorch tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
    ])

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    train_set = OxfordPetsDataset(train_df, transform=transform)
    val_set = OxfordPetsDataset(val_df, transform=transform)
    test_set = OxfordPetsDataset(test_df, transform=transform)

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
    
    wandb.init(
        project="oxford_pet_classification_pre-trained",
        name=f"lr_{learning_rate}_bs_{batch_size}_epochs_{epochs}",
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "scheduler": use_scheduler,
            "total_training_steps": total_training_steps,
            "model_architecture": "ResNet-18",

            "scaled_learning_rate": scale_lr,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon
        }
    )

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
    scheduler = get_scheduler(use_scheduler, optimizer, max_lr=learning_rate,
                              total_steps=total_training_steps, pct_start=0.1, final_div_factor=10)

    criterion = nn.CrossEntropyLoss()

    # record necessary metrics
    global_step = 0
    seen_examples = 0
    best_val_loss = float('inf')

    # Initialize metrics dictionary
    metrics = {
        'train_loss': [],
        'seen_examples': [],
        'val_loss': [],
        'val_accuracy': []
    }

    # training loop
    # for sample in train_loader:
       # print(sample)
       # break  # Just print the first batch to check its structure
    
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
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = optimizer.param_groups[0]['lr']

                pbar.update(inputs.shape[0])
                global_step += 1
                # save necessary metrics in a dictionary; it's recommended to also log seen_examples, which helps you creat appropriate figures in Part 3
                # code here
                metrics['train_loss'].append(loss.item())
                metrics['seen_examples'].append(seen_examples)

                if global_step % batch_steps == 0:
                    # evaluate on validation set
                    val_acc, val_loss, val_loss_step = evaluate(model, val_loader, device)
                    # update metrics from validation results in the dictionary
                    # code here
                    metrics['val_loss'].append(val_loss)
                    metrics['val_accuracy'].append(val_acc)

                    if best_val_loss > val_loss:
                        best_val_loss = val_loss
                        os.makedirs(os.path.join(save_dir, f'{run_name}_{rid}'), exist_ok=True)
                        state_dict = model.state_dict()
                        torch.save(state_dict, os.path.join(save_dir, f'{run_name}_{rid}', 'checkpoint.pth'))
                        print(f'Checkpoint at step {global_step} saved!')
                    # log metrics to wandb
                    # code here
                    wandb.log({"train_loss": loss.item(), "val_loss": val_loss, "val_loss_step" : val_loss_step, "val_accuracy": val_acc, "global_step": global_step, "seen_example": seen_examples, "learning_rate": current_lr})

                pbar.set_postfix(**{'loss (batch)': loss.item()})

    # load best checkpoint and evaluate on test set
    print(f'training finished, run testing using best ckpt...')
    state_dict = torch.load(os.path.join(save_dir, f'{run_name}_{rid}', 'checkpoint.pth'))
    model.load_state_dict(state_dict)
    test_acc, test_loss, test_loss_step = evaluate(model, test_loader, device)

    # log test results to wandb
    # code here
    wandb.log({
        "final_test_accuracy": test_acc,
        "final_test_loss": test_loss,
        "final_test_loss_step": test_loss_step
    })
    wandb.summary.update({
        "final_test_accuracy": test_acc,
        "final_test_loss": test_loss,
        "final_test_loss_step": test_loss_step
    })

    wandb.finish()

# Define the sweep configuration
sweep_config = {
    'method': 'grid',  # Using grid method for fixed learning rates
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'  # Objective is to maximize validation accuracy
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-2, 1e-4, 1e-5, 1e-3]  # Learning rates to test
        },
        'batch_size': {
            'values': [32]
        },
        'epochs': {
            'values': [5]
        },
        'use_scheduler': {
            # 'values': [True, False]
            'values': [False]
        }
    }
}

def sweep_train():
    # Initialize a wandb run for this sweep iteration
    with wandb.init() as run:
        set_seed(42)
        config = run.config  # Access the configuration for this sweep iteration

        # Set up the training environment based on the config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = resnet18(pretrained=False, num_classes=37).to(device)

        # Call the existing training function with parameters from the config
        train_model(
            run_name=run.name,
            model=model,
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            device=device,
            save_dir='./checkpoints',
            use_scheduler=False
        )

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
    '''
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
        use_scheduler=True,
    )
    '''

    # Create a sweep ID and project name
    '''
    rid = random.randint(0, 1000000)
    set_seed(42)
    sweep_id = wandb.sweep(sweep_config, project="oxford_pet_classification_sweep")
    # Use the wandb agent to run the sweep_train function
    wandb.agent(sweep_id, function=sweep_train)
    '''

    # Learning Rate Adjustion (For Adam Optimizer)
    '''
    rid = random.randint(0, 1000000)
    original_batch_size = 32 
    set_seed(42)
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(pretrained=False, num_classes=37)

    # Run experiments with different batch sizes and scaled learning rates
    for batch_size in [original_batch_size // 2, original_batch_size, original_batch_size * 2]:
        for scale_lr in [True, False]:
            train_model(
                run_name=f'bs_{batch_size}_scale_lr_{scale_lr}',
                model=model,
                batch_size=batch_size,
                epochs=args.epochs,
                learning_rate=args.lr,
                device=device,
                save_dir=args.save_dir,
                use_scheduler=False,
                original_batch_size=original_batch_size,
                scale_lr=scale_lr
            )
    '''

    # Using Pretrained Model for fine-tuning tasks.
    rid = random.randint(0, 1000000)
    original_batch_size = 32 
    set_seed(42)
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(pretrained=True)
    # Categories in Oxford Dataset
    num_classes = 37
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Run experiments with different batch sizes and scaled learning rates
    for batch_size in [original_batch_size // 2, original_batch_size]:
        for scale_lr in [True]:
            train_model(
                run_name=f'bs_{batch_size}_scale_lr_{scale_lr}',
                model=model,
                batch_size=batch_size,
                epochs=args.epochs,
                learning_rate=args.lr,
                device=device,
                save_dir=args.save_dir,
                use_scheduler=False,
                original_batch_size=original_batch_size,
                scale_lr=scale_lr
            )
