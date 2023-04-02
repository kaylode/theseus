class DetCompose:
    def __init__(self, transforms, **kwargs):
        self.transforms = transforms

    def __call__(self, image, bboxes, class_labels, **kwargs):
        for t in self.transforms:
            item = t(image=image, bboxes=bboxes, class_labels=class_labels, **kwargs)
            image, bboxes, class_labels = (
                item["image"],
                item["bboxes"],
                item["class_labels"],
            )
        return {"image": image, "bboxes": bboxes, "class_labels": class_labels}
