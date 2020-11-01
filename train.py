from utils.getter import get_instance, seed_everything
import argparse


def main(args, config):
    seed_everything()
    device = torch.device('cuda' if args.cuda is not None else 'cpu')
    
    train_transforms = get_augmentation(config.dataset['augmentations'])
    val_transforms = get_augmentation(config.dataset['augmentations'], types = 'val')

    trainset = get_instance(config.datasets, transforms = train_transforms)
    valset = get_instance(config.datasets, transforms = val_transforms)

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

    backbone = models.resnet101(pretrained=True)
    criterion = get_instance(config.metrics)
    metrics = [get_instance(mcfg) for mcfg in config.metrics]
    
    model = get_instance(
        config.model, 
        num_classes = len(trainset.classes)
        backbone = backbone,
        criterion = criterion,
        metrics = metrics,
        optimizer = torch.optim.Adam,
        optim_params = {'lr': 1e-3},
        device = device)
  
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
                     checkpoint = Checkpoint(save_per_iter=args.save_per_iter, path = args.saved_path),
                     logger = Logger(log_dir = f'loggers/runs/{args.config}'),
                     scheduler = StepLR(model.optimizer, step_size=30, gamma=0.1),
                     evaluate_per_epoch = args.evaluate_per_epoch)
    
    print(trainer)
    trainer.fit(num_epochs=args.num_epochs, print_per_iter=args.print_per_iter, start_epoch=epoch,start_iter=iters)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training custom model')
    parser.add_argument('--cuda', type=bool, default=True, help='Using GPU')
    parser.add_argument('--config', help='yaml config')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--save_per_iter', default=1000, type=int, help='save per iteration')
    parser.add_argument('--evaluate_per_epoch', default=2, type=int, help='evaluate per epoch')
    parser.add_argument('--num_epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--print_per_iter', default=10, type=int,  help='print per iterations')                     
    parser.add_argument('--saved_path', default='weights', type=str, help='save checkpoint to')
    parser.add_argument('--resume', default=None, type=str, help='resume training')                   
    args = parser.parse_args() 
    
    config = Config(os.path.join('configs', args.config + '.yaml'))
    main(args, config)
   