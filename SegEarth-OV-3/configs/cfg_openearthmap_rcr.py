_base_ = './cfg_openearthmap.py'

model = dict(
    model_type='RCR-SegEarth',
    classname_path='./configs/cls_openearthmap.txt',
    confidence_threshold=0.1,
    prob_thd=0.1,
    slide_stride=512,
    slide_crop=512,
    use_rcr=True,
    rcr_config_path='./configs/rcr_openearthmap.yaml',
    rcr_output_dir='outputs/rcr_openearthmap_aux',
    rcr_save_json=False,
)
