def get_model_config(compound_coef=0):
    backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
    fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
    fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
    pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
    anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
    aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    num_scales = len([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    conv_channel_coef = {
        # the channels of P3/P4/P5.
        0: [40, 112, 320],
        1: [40, 112, 320],
        2: [48, 120, 352],
        3: [48, 136, 384],
        4: [56, 160, 448],
        5: [64, 176, 512],
        6: [72, 200, 576],
        7: [72, 200, 576],
        8: [80, 224, 640],
    }


    return {
        "backbone_compound_coef": backbone_compound_coef[compound_coef],
        "fpn_num_filters": fpn_num_filters[compound_coef],
        "fpn_cell_repeats": fpn_cell_repeats[compound_coef],
        "input_sizes": input_sizes[compound_coef],
        "box_class_repeats": box_class_repeats[compound_coef],
        "pyramid_levels": pyramid_levels[compound_coef],
        "anchor_scale": anchor_scale[compound_coef],
        "aspect_ratios": aspect_ratios,
        "num_scales": num_scales,
        "conv_channel_coef": conv_channel_coef[compound_coef],
    }