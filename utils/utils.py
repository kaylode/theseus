import gdown

def download_weights(id_or_url, cached=None, md5=None, quiet=False):
    if id_or_url.startswith('http'):
        url = id_or_url
    else:
        url = 'https://drive.google.com/uc?id={}'.format(id_or_url)

    return gdown.cached_download(url=url, path=cached, md5=md5, quiet=quiet)


weight_url = {
    "yolov5s": "1-2HxtYEqhUtKfUJ-KcnzhHOYGTXq1yFR" ,
    "yolov5m": "1ncet_QWTjHMtfKZwXHuYAXBMfSlyuqU7" ,
    "yolov5l": "1-37KYlOyWiPSjdMuQl9NJi3jH9UvOZpi",
    "yolov5x": "1-4G15VX2jqaUEnXwu8fcNFtYHSe6U0zL",
    "yolov5s6": "1grHwwMPZh51zihsnUaaCivrA9rfs5an3",
    "yolov5m6": "1--FFLutj8WNemE7atcA4B4o-oaIfOQAa",
    "yolov5l6": "1gmU5pn93SMI5ZFFlAr0F2-sELXujyMuL",
    "yolov5x6": "1-0mzWilrNOEaHNvUUW0AoiAoNUPS5hUL",
    "efficientdet-d0": "1-HuqtmQieer05LFqzMCJhzLVlGN7eoaA",
    "efficientdet-d1": "1-H-em-W2p9cKoxjXJHDogoB8aJWwKa7B",
    "efficientdet-d2": "1-SGhC8W6uhN7EaNjueUemqRT2MjQ5D_N",
    "efficientdet-d3": "1-T-8kSzn__csgYVcEDdKOUux86rmAe7d",
    "efficientdet-d4": "1-fvTgw38tzG7XbLxK0k3JD7Z6tP6j2m7",
    "efficientdet-d5": "1-gEU7Y0gFV5S3XW1VviWZXYmhlmcMi-w",
    "efficientdet-d6": "1-hTftFumisKkRwxZ8DHb5gkzC8o_uwpE",
    "efficientdet-d7": "1-lMAUdU8S-5nzu_ubBuaqcFtZc94D_3c",
}

def download_pretrained_weights(name, cached=None):
    return download_weights(weight_url[name], cached)
    
def crop_box(image, box, expand=0):
    #xywh
    h,w,c = image.shape
    # expand box a little
    new_box = box.copy()
    new_box[0] -= expand
    new_box[1] -= expand
    new_box[2] += expand
    new_box[3] += expand

    new_box[0] = max(0, new_box[0])
    new_box[1] = max(0, new_box[1])
    new_box[2] = min(w, new_box[0]+new_box[2])
    new_box[3] = min(h, new_box[1]+new_box[3])

    #xyxy box, cv2 image h,w,c
    return image[int(new_box[1]):int(new_box[3]), int(new_box[0]):int(new_box[2]), :]