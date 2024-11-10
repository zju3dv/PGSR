import os
import json
import argparse

def generate_image_dataset_json(image_dir, test_imgs, output_file):
    all_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    test_images = [img.split(".")[0] for img in test_imgs if img in all_images]
    
    train_images = [img.split(".")[0] for img in all_images if img not in test_images]

    data = {
        "train": train_images,
        "test": test_images
    }
    
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"JSON文件已生成: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="path to dataset folder")

    args = parser.parse_args()

    test_imgs = ["0112.png", "0016.png", "0019.png", "0073.png", "0024.png"]
    generate_image_dataset_json(os.path.join(args.dataset_path, "images"), test_imgs, os.path.join(args.dataset_path, "split.json"))
