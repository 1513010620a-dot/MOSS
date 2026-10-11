import io
import shutil
import tarfile
from pathlib import Path

import scipy.io


SRC = Path("/root/autodl-pub/ImageNet/ILSVRC2012")
OUT = Path("/root/autodl-tmp/imagenet100")
CLASS_FILE = Path("/root/ourpaper/misc/class100.txt")

TRAIN_TAR = SRC / "ILSVRC2012_img_train.tar"
VAL_TAR = SRC / "ILSVRC2012_img_val.tar"
DEVKIT_TAR = SRC / "ILSVRC2012_devkit_t12.tar.gz"

DEVKIT_TMP = Path("/root/autodl-tmp/imagenet_devkit_tmp")


# ============================================================
# Read ImageNet100 class list
# ============================================================
classes = [
    x.strip()
    for x in CLASS_FILE.read_text().splitlines()
    if x.strip()
]

assert len(classes) == 100, \
    f"class100.txt should contain 100 classes, got {len(classes)}"

class_set = set(classes)

print(f"[INFO] ImageNet100 classes: {len(classes)}")


# ============================================================
# Output directories
# ============================================================
train_out = OUT / "train"
val_out = OUT / "val"

train_out.mkdir(parents=True, exist_ok=True)
val_out.mkdir(parents=True, exist_ok=True)

for wnid in classes:
    (train_out / wnid).mkdir(parents=True, exist_ok=True)
    (val_out / wnid).mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Extract ONLY selected 100 training-class archives
#
# Outer train tar:
#   n01440764.tar
#   n01443537.tar
#   ...
#
# Each class tar contains its JPEG images.
# ============================================================
print("\n========== TRAIN ==========")

found = 0

with tarfile.open(TRAIN_TAR, "r") as outer:

    for member in outer:

        name = Path(member.name).name

        if not name.endswith(".tar"):
            continue

        wnid = Path(name).stem

        if wnid not in class_set:
            continue

        found += 1
        print(f"[TRAIN {found:03d}/100] {wnid}")

        class_file = outer.extractfile(member)

        if class_file is None:
            raise RuntimeError(f"Cannot read {member.name}")

        # Streaming mode: do NOT save the inner tar to disk.
        with tarfile.open(fileobj=class_file, mode="r|") as inner:

            for img_member in inner:

                if not img_member.isfile():
                    continue

                img_file = inner.extractfile(img_member)

                if img_file is None:
                    continue

                dst = train_out / wnid / Path(img_member.name).name

                with open(dst, "wb") as f:
                    shutil.copyfileobj(img_file, f)


assert found == 100, \
    f"Expected 100 train classes, but found {found}"

print("[OK] Train extraction finished.")


# ============================================================
# 2. Extract ImageNet devkit metadata
# ============================================================
print("\n========== DEVKIT ==========")

if DEVKIT_TMP.exists():
    shutil.rmtree(DEVKIT_TMP)

DEVKIT_TMP.mkdir(parents=True)

with tarfile.open(DEVKIT_TAR, "r:gz") as tar:
    tar.extractall(DEVKIT_TMP)

meta_hits = list(DEVKIT_TMP.rglob("meta.mat"))
gt_hits = list(
    DEVKIT_TMP.rglob(
        "ILSVRC2012_validation_ground_truth.txt"
    )
)

assert meta_hits, "meta.mat not found"
assert gt_hits, "validation ground truth not found"

meta_path = meta_hits[0]
gt_path = gt_hits[0]

print("meta:", meta_path)
print("gt:", gt_path)


# ============================================================
# 3. ILSVRC ID -> WNID
# ============================================================
meta = scipy.io.loadmat(
    meta_path,
    squeeze_me=True,
    struct_as_record=False
)

synsets = meta["synsets"]

id_to_wnid = {}

for syn in synsets:
    try:
        cls_id = int(syn.ILSVRC2012_ID)
        wnid = str(syn.WNID)

        if 1 <= cls_id <= 1000:
            id_to_wnid[cls_id] = wnid
    except Exception:
        pass

print("[INFO] ImageNet class mappings:", len(id_to_wnid))

assert len(id_to_wnid) == 1000, \
    f"Expected 1000 mappings, got {len(id_to_wnid)}"


# ============================================================
# 4. Validation ground-truth labels
# ============================================================
val_labels = [
    int(x.strip())
    for x in gt_path.read_text().splitlines()
    if x.strip()
]

assert len(val_labels) == 50000, \
    f"Expected 50000 val labels, got {len(val_labels)}"


# ============================================================
# 5. Extract ONLY selected ImageNet100 validation images
# ============================================================
print("\n========== VAL ==========")

selected_val = 0

with tarfile.open(VAL_TAR, "r") as tar:

    for member in tar:

        if not member.isfile():
            continue

        name = Path(member.name).name

        if not name.startswith("ILSVRC2012_val_"):
            continue

        try:
            idx = int(
                name
                .replace("ILSVRC2012_val_", "")
                .replace(".JPEG", "")
            )
        except ValueError:
            continue

        cls_id = val_labels[idx - 1]
        wnid = id_to_wnid[cls_id]

        if wnid not in class_set:
            continue

        img_file = tar.extractfile(member)

        if img_file is None:
            continue

        dst = val_out / wnid / name

        with open(dst, "wb") as f:
            shutil.copyfileobj(img_file, f)

        selected_val += 1

        if selected_val % 500 == 0:
            print(f"[VAL] {selected_val}/5000")


print("[OK] Validation extraction finished:", selected_val)

assert selected_val == 5000, \
    f"Expected 5000 ImageNet100 val images, got {selected_val}"


# ============================================================
# 6. Verification
# ============================================================
print("\n========== VERIFY ==========")

train_classes = [
    p for p in train_out.iterdir()
    if p.is_dir()
]

val_classes = [
    p for p in val_out.iterdir()
    if p.is_dir()
]

train_images = sum(
    1 for p in train_out.rglob("*")
    if p.is_file()
)

val_images = sum(
    1 for p in val_out.rglob("*")
    if p.is_file()
)

print("Train classes:", len(train_classes))
print("Val classes:", len(val_classes))
print("Train images:", train_images)
print("Val images:", val_images)

assert len(train_classes) == 100
assert len(val_classes) == 100
assert val_images == 5000

shutil.rmtree(DEVKIT_TMP, ignore_errors=True)

print("\n====================================")
print("ImageNet100 READY:")
print(OUT)
print("====================================")
