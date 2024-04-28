import numpy as np
import json
import argparse
import pathlib

def convert2llama(root, dst):
    with open(root, 'r') as f:
        test_file = json.load(f)

    output = []
    for scene_id in test_file.keys():
        scene_data = test_file[scene_id]['key_frames']

        for frame_id in scene_data.keys():
            image_paths = scene_data[frame_id]['image_paths']
            assert [image_paths[key].startswith("../nuscenes") for key in image_paths.keys()]
            nuscenes_parent = [str(parent) for parent in root.parents if str(parent).endswith("nuscenes")][0].replace("/nuscenes", "")
            image_paths = [image_paths[key].replace("..", nuscenes_parent) for key in image_paths.keys()]

            frame_data_qa = scene_data[frame_id]['QA']
            QA_pairs = frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + frame_data_qa["behavior"]
            
            for idx, qa in enumerate(QA_pairs):
                question = qa['Q']
                answer = qa['A']
                output.append(
                    {
                        "id": scene_id + "_" + frame_id + "_" + str(idx),
                        "image": image_paths,
                        "conversations": [
                            {
                                "from": "human",
                                "value": "<image>\n" + question
                            },
                            {
                                "from": "gpt",
                                "value": answer
                            },
                        ]
                    }
                )

    with open(dst, 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input-path", type=str, default="/mnt/nlpdata1/home/ismayilz/cs503-project/data/train/nuscenes/v1_1_train_nus_ext.json", help="Input data path")

    args = parser.parse_args()

    root_path = pathlib.Path(args.input_path)
    save_path = root_path.with_name(f"{root_path.stem}_llama{root_path.suffix}")

    convert2llama(root_path, save_path)
