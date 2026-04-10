import numpy as np
from scipy.stats import multivariate_normal
from os import listdir
from os.path import join

# KITTI dataset path and sequences
path = "C:/Neural-Networks-Files/rangevit/kitti_dataset/dataset"
 #'kitti_dataset/dataset'
sequences = ['{:02d}'.format(i) for i in range(11)]

covariance = np.diag(np.array([1, 1, 1]))
center_point = np.zeros((1, 3))

for seq in sequences:
    velo_path = join(path, 'sequences', seq, 'velodyne')
    frames = np.sort([vf[:-4] for vf in listdir(velo_path) if vf.endswith('.bin')])
    seq_path = join(path, 'sequences', seq)
    print('Processing sequence: ' + seq)
    
    for idx, frame in enumerate(frames):
        velo_file = join(seq_path, 'velodyne', frame + '.bin')
        label_file = join(seq_path, 'labels', frame + '.label')
        save_path = join(seq_path, 'labels', frame + '.center')
        
        frame_labels = np.fromfile(label_file, dtype=np.uint32)
        sem_labels = frame_labels & 0xFFFF
        instance_ids = (frame_labels >> 16) & 0xFFFF

        frame_points = np.fromfile(velo_file, dtype=np.float32)
        points = frame_points.reshape((-1, 4))
        center_labels = np.zeros((points.shape[0], 1))

        unique_instances = np.unique(instance_ids)

        for ins in unique_instances:
            if ins == 0:
                continue  # skip background and non-instance points

            valid_ind = np.where(instance_ids == ins)[0]
            if valid_ind.shape[0] < 3:
                continue  # skip small fragments

            sem_classes = np.unique(sem_labels[valid_ind])

            # Optional sanity check for stuff classes
            stuff_classes = [40, 44, 48, 49, 50, 51, 52, 60, 70, 71, 72, 80, 81]
            if any(s in stuff_classes for s in sem_classes):
                print(f"⚠️  Warning: Instance {ins} includes stuff class: {sem_classes}")

            # Compute bounding box center
            x_min, x_max = np.min(points[valid_ind, 0]), np.max(points[valid_ind, 0])
            y_min, y_max = np.min(points[valid_ind, 1]), np.max(points[valid_ind, 1])
            z_min, z_max = np.min(points[valid_ind, 2]), np.max(points[valid_ind, 2])

            center_point[0][0] = (x_min + x_max) / 2
            center_point[0][1] = (y_min + y_max) / 2
            center_point[0][2] = (z_min + z_max) / 2

            # Compute Gaussian score
            gaussians = multivariate_normal.pdf(
                points[valid_ind, 0:3], mean=center_point[0, :3], cov=covariance
            )
            if gaussians.max() != gaussians.min():
                gaussians = (gaussians - gaussians.min()) / (gaussians.max() - gaussians.min())
            else:
                print('error: gaussians max equals min, setting to zero')
                gaussians = np.zeros_like(gaussians)

            center_labels[valid_ind, 0] = gaussians

        np.save(save_path, center_labels)
