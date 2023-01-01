from typing import List, Dict
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

class BoxWithLabel:
    def __init__(self, name, box, label, score) -> None:
        self.name = name
        self.box = box #xmin, xmax, ymin, ymax
        self.label = label
        self.score = score
        self.matched = False
        self.true = False

    def get_box(self):
        return self.box

    def get_label(self):
        return self.label

    def get_score(self):
        return self.score

    def is_matched(self):
        return self.matched

    def set_matched(self):
        self.matched = True

    def is_true(self):
        return self.true

    def set_true(self):
        self.true = True

    def find_best_matching_box(self, boxes, min_iou):
        best_score = 0
        matched_box = None
        for box in boxes:
            if not box.is_matched():
                iou = bb_intersection_over_union(self.get_box(), box.get_box())
                if iou >= min_iou and box.get_score() > best_score:
                    matched_box = box
                    best_score = box.get_score()
        return matched_box, best_score
    

class MatchingPairs:
    def __init__(self, pred_boxes: List[BoxWithLabel], gt_boxes: List[BoxWithLabel], min_iou=0.5, eps=0.000001) -> None:
        self.eps = eps
        self.pred_boxes = pred_boxes
        self.gt_boxes = gt_boxes
        self.min_iou = min_iou

        self.matched = False
        self.pairs = []

    def assign_pair(self, box1, box2, iou):
        box1.set_matched()
        box2.set_matched()
        box1.set_true()
        box2.set_true()
        self.pairs.append([box1, box2, iou])

    def match_pairs(self):
        """
        Loop through ground truth polygons, for each check whether exists prediction match IOU threshold and text
        """
        if len(self.pred_boxes) == 0:
            self.matched = True
            return []
        
        for box in self.gt_boxes:
            # Find predicted boxe that match most
            matched, score = box.find_best_matching_box(self.pred_boxes, self.min_iou)
            if matched is not None:
                matched.set_matched()
                box.set_matched()
                if box.label != matched.label:
                    continue
                # If pred and gt matched completedly, check it
                self.assign_pair(box, matched, score)
        self.matched = True
        return self.pairs

    def get_acc(self):
        tp = self.match_pairs()
        return tp

    def get_false_positive(self):
        if not self.matched:
            self.match_pairs()

        fp = []
        for box in self.pred_boxes:
            if not box.is_true():
                fp.append(box)
        return fp

    def get_false_negative(self):
        if not self.matched:
            self.match_pairs()

        fn = []
        for box in self.gt_boxes:
            if not box.is_true():
                fn.append(box)
        return fn