"""Quick test to verify that v2 transforms jointly transform image + boxes + masks."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

from dataset import OringDefectDataset, get_train_transforms
from config import TRAINING_CONFIG, OUTPUT_ROOT
import torchvision.tv_tensors as tv_tensors

model_dir = OUTPUT_ROOT / "combined"
ann_file = model_dir / "annotations" / "train.json"
img_dir = model_dir / "images" / "train"

# Load without transforms to get baseline
ds_raw = OringDefectDataset(img_dir, ann_file, transforms=None, binary_mode=True)

# Find a defect image
defect_idx = None
for i in range(len(ds_raw)):
    img, tgt = ds_raw[i]
    if tgt["boxes"].shape[0] > 0:
        defect_idx = i
        break

if defect_idx is None:
    print("ERROR: No defect images found in training set!")
    sys.exit(1)

img, tgt = ds_raw[defect_idx]
print(f"Defect image index: {defect_idx}")
print(f"  Image type:  {type(img).__name__}, shape={img.shape}")
print(f"  Boxes type:  {type(tgt['boxes']).__name__}, shape={tgt['boxes'].shape}")
print(f"  Masks type:  {type(tgt['masks']).__name__}, shape={tgt['masks'].shape}")
print(f"  Boxes are tv_tensors.BoundingBoxes: {isinstance(tgt['boxes'], tv_tensors.BoundingBoxes)}")
print(f"  Masks are tv_tensors.Mask:          {isinstance(tgt['masks'], tv_tensors.Mask)}")
print(f"  Image is tv_tensors.Image:          {isinstance(img, tv_tensors.Image)}")
print(f"  Raw boxes: {tgt['boxes']}")
print(f"  Raw mask sum: {tgt['masks'].sum().item()}")

# Load WITH transforms
ds_aug = OringDefectDataset(img_dir, ann_file,
                            transforms=get_train_transforms(TRAINING_CONFIG),
                            binary_mode=True)

print("\nApplying transforms 5 times to same defect image:")
for trial in range(5):
    img_a, tgt_a = ds_aug[defect_idx]
    mask_sum = tgt_a["masks"].sum().item()
    n_boxes = tgt_a["boxes"].shape[0]
    boxes_list = tgt_a["boxes"].tolist() if n_boxes > 0 else []
    boxes_short = str(boxes_list)[:90]
    print(f"  Trial {trial+1}: boxes={n_boxes}, mask_sum={mask_sum:>8}, boxes={boxes_short}")

print("\nDONE - if box coords change across trials, transforms work correctly.")
