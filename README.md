# SHREC'22 track: Pothole and crack detection on road pavement using RGB-D images 


## Descriptions
The aim of this SHREC'22 track is to evaluate the performance of automatic algorithms for the recognition of potholes and cracks on road surfaces. Road safety is one of the top priorities of any public administration as well as being a subject of constant public scrutiny at the local and national levels as road degradation is one of the main causes of accidents. Currently, the scheduling of inspections and maintenance is entrusted to specialized personnel who require specific training and operate expensive and bulky machinery. This proposal aims to automate the detection of road deterioration by enabling timely monitoring of large areas of road pavement through the use of the latest Deep Learning techniques. The goal is to segment and recognize images and videos, using a training set generated with RGB-D images. The track is organized by IMATI CNR .

## Dataset, ground truth and evaluation
Participants have at their disposal about 4k pairs of images (made of RGB+segmentation masks), taken from public, high quality datasets. In addition to these images, we provide RGB-D video clips from which participants can extract additional images to enrich the dataset. The disparity map of these videos is noisy and needs denoising before it can become a true segmentation mask. So (as often happens in real ML problems) training skills are not the only ones to determine ultimate success. The final aim of the task is to train neural network models capable of performing the semantic segmentation of road surface damage (potholes and cracks). The quality of the trained models will be evaluated both on a test set (about 500 RGB + masks pairs) held out and not shared with the participants, and on real videos captured "in the wild". The evaluation will be of two types: quantitative and qualitative, each with a 50% weight on the final evaluation score. The quantitative assessment will be based on standard metrics such as Dice multiclass and "weighted" pixel accuracy. The qualitative evaluation will be at the sole discretion of the panel of organizers of the challenge, which will evaluate (motivating them) the visual accuracy of the segmentation, its temporal stability, amount of false positives, false negatives, etc. On the off chance that two participants get the same final score (quantitative + qualitative), we will reward the network capable of inference with the highest FPS, and secondly, with the lowest memory footprint/img size. Each participant is allowed to send us up to 3 outcomes for each task.
Registration and instructions for competitors
Each participant is requested to register to the track by sending an email to Elia Moscoso Thompson (email: elia.moscoso@ge.imati.cnr.it) with the subject SHREC'22 track: Pothole and crack detection on road pavement using RGB-D images. Then, an answer will be sent to each participant with further instructions on how to download the models once the constest starts.

## Further information

To maximize the reproducibility of the experiments, participants are required to share with the organizers the trained models, the source code with which they trained the models, and a portion of standalone code that can be used to perform inference on an image or video.
## Important dates [UPDATED - Jan 31th 2022]

- February 1, 2022: a set of samples is released.
- February 14-28, 2022: the dataset is available and the participants start to run their methods.
- March 1, 2022: the participants submit up to 3 runs to the organizers and send to the organizers a summary of their method(s).
- March 5, 2022: an abstract of the report is submitted to the participants.
- March 15, 2022: submission of the track full report to the SHREC organizers.
