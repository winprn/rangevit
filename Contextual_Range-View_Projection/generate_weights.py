import numpy as np
import yaml
from os import listdir
from os.path import join


# === Define weights ===
SEMANTICKITTI_WEIGHTS = np.array([
    0.0,   #  0 - ignored
    0.0,   #  1 - car
    0.0,  #  2 - bicycle
    -1.0,  #  3 - motorcycle
    -1.0,   #  4 - truck
    0.0,   #  5 - other-vehicle
    0.0,  #  6 - person
    -1.0,  #  7 - bicyclist
    0.0,  #  8 - motorcyclist
    0.0,   #  9 - road
    0.0,   # 10 - parking
    0.0,   # 11 - sidewalk
    0.0,   # 12 - other-ground
    0.0,   # 13 - building
    0.0,   # 14 - fence
    0.0,   # 15 - vegetation
    0.0,   # 16 - trunk
    0.0,   # 17 - terrain
    0.0,   # 18 - pole
    0.0    # 19 - traffic sign
])


def load_sequences():
    path = "C:/Neural-Networks-Files/rangevit/kitti_dataset/dataset" #'kitti_dataset/dataset'
    config_file = join(path, 'semantic-kitti.yaml')
    with open(config_file) as f:
        config = yaml.safe_load(f)

    learning_map = config['learning_map']
    max_key = max(learning_map.keys())
    # +100 hack to cover unknown labels
    class_map_lut = np.zeros((max_key + 100), dtype=np.int32)
    for k, v in learning_map.items():
        class_map_lut[k] = v

    sequences = ['{:02d}'.format(i) for i in range(11)]
    return path, sequences, class_map_lut


def process_sequence(path, seq, class_weights):
    print(f'Processing sequence: {seq}')
    seq_path = join(path, 'sequences', seq)
    label_folder = join(seq_path, 'labels')

    _, _, class_map_lut = load_sequences()

    frames = sorted([f.replace('.label', '') for f in listdir(label_folder) if f.endswith('.label')])
    for frame in frames:
        label_file = join(label_folder, frame + '.label')
        weight_file = join(label_folder, frame + '.weight.npy')

        labels = np.fromfile(label_file, dtype=np.uint32)
        sem_labels = labels & 0xFFFF  # Extract semantic class
        sem_labels = class_map_lut[sem_labels]

        if sem_labels.max() >= len(class_weights):
            raise ValueError(f"Found semantic label {sem_labels.max()} which exceeds available class weights")

        # Apply weights
        weights = class_weights[sem_labels]
        np.save(weight_file, weights)


def main():
    path, sequences, _ = load_sequences()
    class_weights = SEMANTICKITTI_WEIGHTS

    for seq in sequences:
        process_sequence(path, seq, class_weights)


if __name__ == '__main__':
    main()
