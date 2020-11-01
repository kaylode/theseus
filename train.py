from utils.getter import *
import argparse
import pandas as pd

def main(args, config):

    device = torch.device('cuda' if args.cuda is not None else 'cpu')
    
    
    train_transforms = get_augmentation(config)
    val_transforms = get_augmentation(config, types = 'val')


    trainset = ImageClassificationDataset(
        img_dir = config.dataset.img_dir,
        transforms = train_transforms,
        max_samples = None)

    valset = ImageClassificationDataset(
        img_dir = config.dataset.img_dir,
        transforms = val_transforms,
        max_samples = None)

    print(trainset)
    print(valset)
 
    trainloader = data.DataLoader(
        trainset, 
        batch_size=args.batch_size, 
        collate_fn=trainset.collate_fn,
        num_workers=4,
        pin_memory=True, 
        shuffle=True)

    valloader = data.DataLoader(
        valset, 
        batch_size=args.batch_size, 
        collate_fn=valset.collate_fn,
        num_workers=4,
        pin_memory=True, 
        shuffle=False)
    
    NUM_CLASSES = len(config.classes)

    backbone = models.resnet101(pretrained=True)
    model = Classifier(
        backbone= backbone,
        criterion = smoothCELoss(),
        metrics= AccuracyMetric(),
        optimizer= torch.optim.Adam,
        optim_params = {'lr': 1e-3},
        device = device
    )
    
    model.model_name = args.config

    if args.resume is not None:
        load_checkpoint(model, args.resume)
        epoch, iters = get_epoch_iters(args.resume)
    else:
        epoch = 0
        iters = 0

    trainer = Trainer(model,
                     trainloader, 
                     valloader,
                     checkpoint = Checkpoint(save_per_iter=1000, path = args.saved_path),
                     logger = Logger(log_dir = f'loggers/runs/{args.config}'),
                     scheduler = StepLR(model.optimizer, step_size=30, gamma=0.1),
                     evaluate_per_epoch = 2)
    
    print(trainer)
    trainer.fit(num_epochs=50, print_per_iter=10,start_epoch=epoch,start_iter=iters)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training on Flickr 30k')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='Using GPU')
    parser.add_argument('--config', default='dualpath',
                        help='yaml config')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--saved_path', default='weights', type=str,
                        help='save checkpoint to')
    parser.add_argument('--resume', default=None, type=str,
                        help='resume training')                   
    args = parser.parse_args() 
    
    config = Config(os.path.join('configs', args.config + '.yaml'))
    main(args, config)
   