import py_vncorenlp
from .base import BaseProcessor

class VNCoreNLPProcessor(BaseProcessor):
    def __init__(self, save_dir:str='/mnt/4TBSSD/zalo/e2e_qa'):
        py_vncorenlp.download_model(save_dir=save_dir)
        # Load the word and sentence segmentation component
        self.rdrsegmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"], 
            save_dir=save_dir
        )

    def __call__(self, text):
        text = self.rdrsegmenter.word_segment(text)
        return text
