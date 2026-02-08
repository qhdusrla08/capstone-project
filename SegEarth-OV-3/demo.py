from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from mmseg.structures import SegDataSample
from segearthov3_segmentor import SegEarthOV3Segmentation

img_path = 'resources/oem_koeln_50.tif'

name_list = ['background', 'bareland,barren', 'grass', 'road', 'car',
             'tree,forest', 'water,river', 'cropland', 'building,roof,house']

with open('./configs/my_name.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list)-1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')
writers.close()


img = Image.open(img_path)
img_tensor = transforms.Compose([
    transforms.ToTensor(),
])(img).unsqueeze(0).to('cuda') # This variable is only a placeholder; the actual data is read within the model. (To be optimized)

data_sample = SegDataSample()
img_meta = {
    'img_path': img_path,
    'ori_shape': img.size[::-1] # H, W
}
data_sample.set_metainfo(img_meta)


model = SegEarthOV3Segmentation(
    type='SegEarthOV3Segmentation',
    model_type='SAM3',
    classname_path='./configs/my_name.txt',
    prob_thd=0.1,
    confidence_threshold=0.1,
    slide_stride=512,
    slide_crop=512,
)

seg_pred = model.predict(img_tensor, data_samples=[data_sample])
seg_pred = seg_pred[0].pred_sem_seg.data.cpu().numpy().squeeze(0)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].axis('off')
ax[1].imshow(seg_pred, cmap='viridis')
ax[1].axis('off')
plt.tight_layout()
# plt.show()
plt.savefig('seg_pred.png', bbox_inches='tight')
