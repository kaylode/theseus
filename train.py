from utils.getter import *
import argparse
import os


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
seed_everything()

def train(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    num_gpus = len(config.gpu_devices.split(','))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    trainset, valset, trainloader, valloader = get_dataset_and_dataloader(config)
  
    net = BaseTimmModel(
        name=config.model_name, 
        num_classes=trainset.num_classes)

    if args.saved_path is not None:
        args.saved_path = os.path.join(args.saved_path, config.project_name)

    if args.log_path is not None:
        args.log_path = os.path.join(args.log_path, config.project_name)

    # if config.tta:
    #     config.tta = TTA(
    #         min_conf=config.min_conf_val, 
    #         min_iou=config.min_iou_val, 
    #         postprocess_mode=config.tta_ensemble_mode)
    # else:
    #     config.tta = None

    metric = [
        AccuracyMetric(),
        BalancedAccuracyMetric(num_classes=trainset.num_classes), 
        ConfusionMatrix(trainset.classes), 
        F1ScoreMetric(n_classes=trainset.num_classes)
    ]

    optimizer, optimizer_params = get_lr_policy(config.lr_policy)

    if config.mixed_precision:
        scaler = NativeScaler()
    else:
        scaler = None

    model = Classifier(
            model = net,
            metrics=metric,
            scaler=scaler,
            criterion=nn.CrossEntropyLoss(),
            optimizer= optimizer,
            optim_params = optimizer_params,     
            device = device)

    if args.resume is not None:                
        load_checkpoint(model, args.resume)
        start_epoch, start_iter, best_value = get_epoch_iters(args.resume)
    else:
        print('Not resume. Initialize weights')
        start_epoch, start_iter, best_value = 0, 0, 0.0
        
    scheduler, step_per_epoch = get_lr_scheduler(
        model.optimizer, 
        lr_config=config.lr_scheduler,
        num_epochs=config.num_epochs)

    trainer = Trainer(config,
                     model,
                     trainloader, 
                     valloader,
                     checkpoint = Checkpoint(save_per_iter=args.save_interval, path = args.saved_path),
                     best_value=best_value,
                     logger = Logger(log_dir=args.log_path),
                     scheduler = scheduler,
                     visualize_when_val = args.gradcam_visualization,
                     evaluate_per_epoch = args.val_interval,
                     step_per_epoch = step_per_epoch)
    
    print("##########   DATASET INFO   ##########")
    print("Trainset: ")
    print(trainset)
    print("Valset: ")
    print(valset)
    print()
    print(trainer)
    print()
    print(config)
    print(f'Training with {num_gpus} gpu(s)')
    print(f"Start training at [{start_epoch}|{start_iter}]")
    print(f"Current best MAP: {best_value}")
    
    trainer.fit(start_epoch = start_epoch, start_iter = start_iter, num_epochs=config.num_epochs, print_per_iter=args.print_per_iter)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training EfficientDet')
    parser.add_argument('config' , default='config', type=str, help='project file that contains parameters')
    parser.add_argument('--print_per_iter', type=int, default=300, help='Number of iteration to print')
    parser.add_argument('--val_interval', type=int, default=2, help='Number of epoches between valing phases')
    parser.add_argument('--gradcam_visualization', action='store_true', help='whether to visualize box to ./sample when validating (for debug), default=off')
    parser.add_argument('--save_interval', type=int, default=1000, help='Number of steps between saving')
    parser.add_argument('--log_path', type=str, default='loggers/runs')
    parser.add_argument('--resume', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize')
    parser.add_argument('--saved_path', type=str, default='./weights')
    parser.add_argument('--freeze_backbone', action='store_true', help='whether to freeze the backbone')
    
    args = parser.parse_args()
    config = Config(os.path.join('configs','configs.yaml'))

    train(args, config)
    


