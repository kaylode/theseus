from utils.getter import *
from models.retinanet import *
from models.ssd import *

transforms = Compose([
    Resize((300,300)),
    ToTensor(),
    Normalize(),
])

if __name__ == "__main__":
    dataset_path = "datasets/datasets/VOC/"
    img_path = dataset_path + "images"
    ann_path = {
        "train": dataset_path + "annotations/pascal_train2012.json",
        "val": dataset_path + "annotations/pascal_val2012.json"}
   
    trainset = ObjectDetectionDataset(img_dir=img_path, ann_path = ann_path['train'],transforms= transforms)
    valset = ObjectDetectionDataset(img_dir=img_path, ann_path = ann_path['val'],transforms= transforms)
    print(trainset)
    print(valset)

    
    NUM_CLASSES = len(trainset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    # Dataloader
    BATCH_SIZE = 4
    my_collate = trainset.collate_fn
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=True)
    valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=False)
    
    criterion = MultiBoxLoss
    optimizer = torch.optim.SGD
    #metrics = [AccuracyMetric(decimals=3)]
    
    model = RetinaDetector(
                    n_classes = NUM_CLASSES,
                    optim_params = {'lr': 1e-3, 'momentum': 0.9},
                    criterion= criterion, 
                    optimizer= optimizer,
                    #metrics=  metrics,
                    device = device)
    
    #load_checkpoint(model, "weights/2020-08-25_04-41-25/RetinaNet-29.pth")
    model.unfreeze()
    trainer = Trainer(model,
                     trainloader, 
                     valloader,
#                     clip_grad = 1.0,
                     checkpoint = Checkpoint(save_per_epoch=5),
                     logger = Logger(log_dir='loggers/runs/retina2'),
                     scheduler = StepLR(model.optimizer, step_size=20, gamma=0.1),
                     evaluate_per_epoch = 0)
    
    print(trainer)
    
    """results = trainer.inference_batch(valloader)
    print(results)
    print(valset.classes[results['labels'][3].cpu().numpy()[1]])
    valset.visualize_item(3,figsize=(5,5))"""
    
    trainer.fit(num_epochs=30, print_per_iter=10)
    

  
