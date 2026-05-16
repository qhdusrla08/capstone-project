_base_ = './cfg_vaihingen.py'

model = dict(
    model_type='RCR-SegEarth',
    classname_path='./configs/cls_vaihingen.txt',
    confidence_threshold=0.4,
    prob_thd=0.1,
    bg_idx=5,
    use_rcr=True,
    rcr_config_path='./configs/rcr_vaihingen.yaml',
    rcr_output_dir='outputs/rcr_vaihingen_aux',
    rcr_save_json=False,
)
