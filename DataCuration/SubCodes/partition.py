import os
import shutil

SOURCE_DIR = "./sampled_output"
DEST_DIR = "./Partition"
NUM_BATCHES = 3  # part_1, part_2, part_3
CATEGORIES_PER_BATCH = 192  # 576 / 3

def split_folders():
    # Get all first-level category folders
    all_categories = sorted([
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ])

    total_categories = len(all_categories)
    assert total_categories == CATEGORIES_PER_BATCH * NUM_BATCHES, f"Expected {CATEGORIES_PER_BATCH * NUM_BATCHES} categories, found {total_categories}"

    print(f"Splitting {total_categories} categories into {NUM_BATCHES} parts...")

    for i in range(NUM_BATCHES):
        part_name = f"part_{i+1}"
        part_path = os.path.join(DEST_DIR, part_name)
        os.makedirs(part_path, exist_ok=True)

        start = i * CATEGORIES_PER_BATCH
        end = start + CATEGORIES_PER_BATCH
        categories = all_categories[start:end]

        for cat in categories:
            src = os.path.join(SOURCE_DIR, cat)
            dest = os.path.join(part_path, cat)
            shutil.copytree(src, dest)

        print(f"✅ {part_name} created with {len(categories)} categories.")

    print("🎉 All batches created successfully under ./Batches")

if __name__ == "__main__":
    split_folders()
