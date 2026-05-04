import shutil
from pathlib import Path

ROOT = Path("data/datasets")
LEGACY = ROOT / "legacy"
OUT = ROOT / "cicadas_merged"

MERGED_NAMES = [
    "cannon",
    "evo_cannon",
    "evo_ice_spirit",
    "evo_musketeer",
    "evo_skeletons",
    "fireball",
    "hero_ice_golem",
    "hero_musketeer",
    "hog_rider",
    "ice_golem",
    "ice_spirit",
    "log",
    "musketeer",
    "skeletons",
]

# cicadas (legacy, 10 classes):
# cannon,evo_ice_spirit,evo_skeletons,fireball,hog_rider,ice_golem,ice_spirit,log,musketeer,skeletons
CICADAS_MAP = {0: 0, 1: 2, 2: 4, 3: 5, 4: 8, 5: 9, 6: 10, 7: 11, 8: 12, 9: 13}

# torchroyale-2 (datasets, 13 classes — no evo-skeletons this time):
# cannon,evo-cannon,evo-musketeer,evo-skeleton,fireball,
# hero-ice-golem,hero-musketeer,hog-rider,ice-golem,ice-spirit,log,musketeer,skeleton
TR2_MAP = {
    0: 0,
    1: 1,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
}


def remap_label(src, dst, class_map):
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in src.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        parts[0] = str(class_map[int(parts[0])])
        lines.append(" ".join(parts))
    dst.write_text("\n".join(lines) + "\n")


def merge_split(dataset_path, class_map, split):
    img_src = dataset_path / split / "images"
    lbl_src = dataset_path / split / "labels"
    img_dst = OUT / split / "images"
    lbl_dst = OUT / split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    prefix = dataset_path.name + "_"
    count = 0
    for img in img_src.glob("*"):
        shutil.copy2(img, img_dst / (prefix + img.name))
        lbl = lbl_src / (img.stem + ".txt")
        if lbl.exists():
            remap_label(lbl, lbl_dst / (prefix + img.stem + ".txt"), class_map)
        count += 1
    print(f"  {dataset_path.name}/{split}: {count} images")
    return count


if OUT.exists():
    shutil.rmtree(OUT)

total_train = 0
for split in ("train", "valid", "test"):
    print(f"-- {split} --")
    n = merge_split(LEGACY / "cicadas", CICADAS_MAP, split)
    m = merge_split(ROOT / "torchroyale-2", TR2_MAP, split)
    if split == "train":
        total_train = n + m

(OUT / "data.yaml").write_text(
    f"train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n"
    f"nc: {len(MERGED_NAMES)}\nnames: {MERGED_NAMES}\n"
)

# Count annotations per class
counts = {n: 0 for n in MERGED_NAMES}
for lbl in (OUT / "train" / "labels").glob("*.txt"):
    for line in lbl.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            counts[MERGED_NAMES[int(parts[0])]] += 1

print(f"\nTrain: {total_train} images")
print(f"Valid: {len(list((OUT / 'valid' / 'images').glob('*')))}")
print("Annotations per class (train):")
for n, c in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {n:<25} {c:>5}")

# Move torchroyale-2 to legacy as torchroyale-hog2.6
old_legacy = LEGACY / "torchroyale-2"
if old_legacy.exists():
    shutil.rmtree(old_legacy)
    print("\nRemoved legacy/torchroyale-2")
shutil.move(str(ROOT / "torchroyale-2"), str(LEGACY / "torchroyale-hog2.6"))
print("Moved torchroyale-2 → legacy/torchroyale-hog2.6")
