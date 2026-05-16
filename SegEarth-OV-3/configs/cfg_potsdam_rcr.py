_base_ = './cfg_potsdam.py'

model = dict(
    model_type='RCR-SegEarth',
    classname_path='./configs/cls_potsdam.txt',
    confidence_threshold=0.2,
    prob_thd=0.1,
    bg_idx=5,
    use_rcr=True,
    rcr_config_path='./configs/rcr_potsdam.yaml',
    rcr_output_dir='outputs/rcr_potsdam_aux',
    rcr_save_json=False,
)
