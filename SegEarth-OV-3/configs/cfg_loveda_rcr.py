_base_ = './cfg_loveda.py'

model = dict(
    model_type='RCR-SegEarth',
    classname_path='./configs/cls_loveda.txt',
    confidence_threshold=0.5,
    prob_thd=0.5,
    use_rcr=True,
    rcr_config_path='./configs/rcr_loveda.yaml',
    rcr_output_dir='outputs/rcr_loveda_aux',
    rcr_save_json=False,
)
