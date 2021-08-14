from utils.getter import *
import argparse

parser = argparse.ArgumentParser('Training Object Detection')
parser.add_argument('--print_per_iter', type=int, default=300, help='Number of iteration to print')
parser.add_argument('--val_interval', type=int, default=2, help='Number of epoches between valing phases')
parser.add_argument('--save_interval', type=int, default=1000, help='Number of steps between saving')
parser.add_argument('--resume', type=str, default=None,
                    help='whether to load weights from a checkpoint, set None to initialize')
parser.add_argument('--saved_path', type=str, default='./weights')
parser.add_argument('--no_visualization', action='store_false', help='whether to visualize box to ./sample when validating (for debug), default=on')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
seed_everything()

def train(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    num_gpus = len(config.gpu_devices.split(','))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    devices_info = get_devices_info(config.gpu_devices)
    
    trainset, valset, trainloader, valloader = get_dataset_and_dataloader(config)
    
    net = get_transformer_model(
        src_vocab=trainloader.src_tokenizer.vocab_size,
        trg_vocab=trainloader.tgt_tokenizer.vocab_size)

    optimizer, optimizer_params = get_lr_policy(config.lr_policy)

    criterion = nn.CrossEntropyLoss(
            ignore_index=trainset.tgt_tokenizer.pad_token_id)

    model = Translator(
            model = net,
            criterion=criterion,
            metrics=None,
            scaler=NativeScaler(),
            optimizer= optimizer,
            optim_params = optimizer_params,     
            device = device)

    if args.resume is not None:                
        load_checkpoint(model, args.resume)
        start_epoch, start_iter, best_value = get_epoch_iters(args.resume)
    else:
        print('Not resume. Load pretrained weights...')
        # args.resume = os.path.join(CACHE_DIR, f'{config.model_name}.pth')
        # download_pretrained_weights(f'{config.model_name}', args.resume)
        # load_checkpoint(model, args.resume)
        start_epoch, start_iter, best_value = 0, 0, 0.0
        
    scheduler, step_per_epoch = get_lr_scheduler(
        model.optimizer, 
        lr_config=config.lr_scheduler,
        num_epochs=config.num_epochs)

    if args.resume is not None:                 
        old_log = find_old_log(args.resume)
    else:
        old_log = None

    args.saved_path = os.path.join(
        args.saved_path, 
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    trainer = Trainer(config,
                     model,
                     trainloader, 
                     valloader,
                     checkpoint = Checkpoint(save_per_iter=args.save_interval, path = args.saved_path),
                     best_value=best_value,
                     logger = Logger(log_dir=args.saved_path, resume=old_log),
                     scheduler = scheduler,
                     evaluate_per_epoch = args.val_interval,
                     visualize_when_val = args.no_visualization,
                     step_per_epoch = step_per_epoch)
    print()
    print("##########   DATASET INFO   ##########")
    print("Trainset: ")
    print(trainset)
    print("Valset: ")
    print(valset)
    print()
    print(trainer)
    print()
    print(config)
    print(f'Training with {num_gpus} gpu(s): ')
    print(devices_info)
    print(f"Start training at [{start_epoch}|{start_iter}]")
    
    trainer.fit(start_epoch = start_epoch, start_iter = start_iter, num_epochs=config.num_epochs, print_per_iter=args.print_per_iter)

    

if __name__ == '__main__':
    
    args = parser.parse_args()
    config = Config(os.path.join('configs','config.yaml'))

    train(args, config)