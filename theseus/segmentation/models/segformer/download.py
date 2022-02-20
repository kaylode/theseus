import torch
from theseus.utilities.download import download_from_drive

pretrained_urls = {
    'MiT-B1': "18PN_P3ajcJi_5Q2v8b4BP9O4VdNCpt6m",
    'MiT-B2': "1AcgEK5aWMJzpe8tsfauqhragR0nBHyPh",
    "MiT-B3": "1-OmW3xRD3WAbJTzktPC-VMOF5WMsN8XT"
}

def load_pretrained(model, name):
    if name in pretrained_urls.keys():
        pretrained_id = pretrained_urls[name]
        filepath = download_from_drive(
            pretrained_id, 
            f"{name}.pth", 
            cache=True)

        state_dict = torch.load(filepath)
        model.load_state_dict(state_dict)

    return model
