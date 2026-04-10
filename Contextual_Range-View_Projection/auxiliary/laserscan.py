#!/usr/bin/env python3
import numpy as np
import os


class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, scan_proj=False, use_center=False, use_weight=False):
    self.project = project
    self.proj_H = H
    self.proj_W = W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down
    self.scan_proj = scan_proj
    self.use_center = use_center
    self.use_weight = use_weight
    self.center_scores = None
    self.weight_scores = None
    self.reset()

  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)

    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                            dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                            dtype=np.int32)

    # for each point, where it is in the range image
    self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                              dtype=np.int32)       # [H,W] mask

  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  # def open_scan(self, filename, center_file='', weight_file=''):
  #   """ Open raw scan and fill in attributes
  #   """
  #   # reset just in case there was an open structure
  #   self.reset()

  #   # check filename is string
  #   if not isinstance(filename, str):
  #     raise TypeError("Filename should be string type, "
  #                     "but was {type}".format(type=str(type(filename))))

  #   # check extension is a laserscan
  #   if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
  #     raise RuntimeError("Filename extension is not valid scan file.")

  #   # if all goes well, open pointcloud
  #   scan = np.fromfile(filename, dtype=np.float32)
  #   scan = scan.reshape((-1, 4))

  #   # put in attribute
  #   points = scan[:, 0:3]    # get xyz
  #   remissions = scan[:, 3]  # get remission
  #   self.set_points(points, remissions)

  def open_scan(self, filename, center_file='', weight_file=''):
    """ Open raw scan and fill in attributes, optionally loading center/weight scores """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
        raise TypeError(f"Filename should be string, got {type(filename)}")

    # check extension is valid
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
        raise RuntimeError("Filename extension is not a valid scan file.")

    # read point cloud
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, 0:3]
    remissions = scan[:, 3]

    # optionally load .center.npy and .weight.npy
    if center_file and os.path.isfile(center_file):
        self.center_scores = np.load(center_file)
        # print('loading center file')
    else:
        self.center_scores = None

    if weight_file and os.path.isfile(weight_file):
        self.weight_scores = np.load(weight_file)
    else:
        self.weight_scores = None

    # set points and remissions
    self.set_points(points, remissions)


  def set_points(self, points, remissions=None):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    # put in attribute
    self.points = points    # get xyz
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

    # if projection is wanted, then do it and fill in the structure
    if self.project and self.scan_proj:
      self.do_scan_projection()
    else:
      self.do_range_projection()

  def do_range_projection(self):
    """ Project a pointcloud into a spherical projection image """
    # Laser parameters
    fov_up = self.proj_fov_up / 180.0 * np.pi
    fov_down = self.proj_fov_down / 180.0 * np.pi
    fov = abs(fov_down) + abs(fov_up)

    # Get depth of all points
    depth = np.linalg.norm(self.points, 2, axis=1)

    # Get scan components
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]

    # Get angles
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / (depth + 1e-8))

    # Project into image coordinates
    proj_x = 0.5 * (yaw / np.pi + 1.0) * self.proj_W
    proj_y = (1.0 - (pitch + abs(fov_down)) / fov) * self.proj_H

    # Clamp and floor
    proj_x = np.clip(np.floor(proj_x), 0, self.proj_W - 1).astype(np.int32)
    proj_y = np.clip(np.floor(proj_y), 0, self.proj_H - 1).astype(np.int32)

    self.proj_x = np.copy(proj_x)
    self.proj_y = np.copy(proj_y)
    self.unproj_range = np.copy(depth)

    # Determine sorting criterion
    if self.use_center and self.center_scores is not None:
        sorting_criterion = depth * (1.0 / (self.center_scores.squeeze() + 0.01))
    elif self.use_weight and self.weight_scores is not None:
        sorting_criterion = depth * (1.0 / (self.weight_scores.squeeze() + 0.01))
    else:
        sorting_criterion = depth

    # Sort by decreasing score
    indices = np.arange(depth.shape[0])
    order = np.argsort(sorting_criterion)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]
    proj_x = proj_x[order]
    proj_y = proj_y[order]

    # Fill projection outputs
    self.proj_range[proj_y, proj_x] = depth
    self.proj_xyz[proj_y, proj_x] = points
    self.proj_remission[proj_y, proj_x] = remission
    self.proj_idx[proj_y, proj_x] = indices
    self.proj_mask = (self.proj_idx >= 0).astype(np.float32)



  def do_scan_projection(self):
    """ Project a pointcloud using custom scan-based projection logic """
    depth = np.linalg.norm(self.points, 2, axis=1)
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]

    yaw = -np.arctan2(scan_y, -scan_x)
    proj_x = 0.5 * (yaw / np.pi + 1.0)

    # Scan line separation logic
    new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
    proj_y = np.zeros_like(proj_x)
    proj_y[new_raw] = 1
    proj_y = np.cumsum(proj_y)

    proj_x = proj_x * self.proj_W - 0.001

    proj_x = np.clip(np.floor(proj_x), 0, self.proj_W - 1).astype(np.int32)
    proj_y = np.clip(np.floor(proj_y), 0, self.proj_H - 1).astype(np.int32)

    self.proj_x = np.copy(proj_x)
    self.proj_y = np.copy(proj_y)
    self.unproj_range = np.copy(depth)

    # Determine sorting criterion
    if self.use_center and self.center_scores is not None:
        # print(max(self.center_scores.squeeze()))
        # print((-self.center_scores.squeeze()) + 0.00001)
        sorting_criterion = depth * (1.0 / (self.center_scores.squeeze() + 0.01)) # (1.0/(self.center_scores+0.01).squeeze(1)) #
        # print(sorting_criterion)
    elif self.use_weight and self.weight_scores is not None:
        sorting_criterion = depth * (1.0 / (self.weight_scores.squeeze() + 0.01))
    else:
        sorting_criterion = depth

    # Sort by decreasing score
    indices = np.arange(depth.shape[0])
    order = np.argsort(sorting_criterion)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]
    proj_x = proj_x[order]
    proj_y = proj_y[order]

    # Fill projection outputs
    self.proj_range[proj_y, proj_x] = depth
    self.proj_xyz[proj_y, proj_x] = points
    self.proj_remission[proj_y, proj_x] = remission
    self.proj_idx[proj_y, proj_x] = indices
    self.proj_mask = (self.proj_idx >= 0).astype(np.float32)



class SemLaserScan(LaserScan):
  """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
  EXTENSIONS_LABEL = ['.label']

  def __init__(self, sem_color_dict=None, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, scan_proj=False, use_center=False, use_weight=False):
    super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down, scan_proj, use_center, use_weight)
    self.reset()

    # make semantic colors
    max_sem_key = 0
    for key, data in sem_color_dict.items():
      if key + 1 > max_sem_key:
        max_sem_key = key + 1
    self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
    for key, value in sem_color_dict.items():
      self.sem_color_lut[key] = np.array(value, np.float32) / 255.0

    # make instance colors
    max_inst_id = 100000
    self.inst_color_lut = np.random.uniform(low=0.0,
                                            high=1.0,
                                            size=(max_inst_id, 3))
    # force zero to a gray-ish color
    self.inst_color_lut[0] = np.full((3), 0.1)

  def reset(self):
    """ Reset scan members. """
    super(SemLaserScan, self).reset()

    # semantic labels
    self.sem_label = np.zeros((0, 1), dtype=np.uint32)         # [m, 1]: label
    self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # instance labels
    self.inst_label = np.zeros((0, 1), dtype=np.uint32)         # [m, 1]: label
    self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # projection color with semantic labels
    self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                   dtype=np.int32)              # [H,W]  label
    self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                   dtype=float)              # [H,W,3] color

    # projection color with instance labels
    self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                                    dtype=np.int32)              # [H,W]  label
    self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
                                    dtype=float)              # [H,W,3] color
    # ------------ added new
    self.sem_label_proj = self.proj_sem_label
    # ------------- end added

  def open_label(self, filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
      raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))

    # set it
    self.set_label(label)

  def set_label(self, label):
    """ Set points for label not from file but from np
    """
    # check label makes sense
    if not isinstance(label, np.ndarray):
      raise TypeError("Label should be numpy array")

    # only fill in attribute if the right size
    if label.shape[0] == self.points.shape[0]:
      self.sem_label = label & 0xFFFF  # semantic label in lower half
      self.inst_label = label >> 16    # instance id in upper half
    else:
      print("Points shape: ", self.points.shape)
      print("Label shape: ", label.shape)
      raise ValueError("Scan and Label don't contain same number of points")

    # sanity check
    assert((self.sem_label + (self.inst_label << 16) == label).all())

    if self.project:
      self.do_label_projection()

  def colorize(self):
    """ Colorize pointcloud with the color of each semantic label
    """
    self.sem_label_color = self.sem_color_lut[self.sem_label]
    self.sem_label_color = self.sem_label_color.reshape((-1, 3))

    self.inst_label_color = self.inst_color_lut[self.inst_label]
    self.inst_label_color = self.inst_label_color.reshape((-1, 3))

  def do_label_projection(self):
    # only map colors to labels that exist
    mask = self.proj_idx >= 0

    # semantics
    self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
    self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]

    # instances
    self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
    self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]

    # ✅ Make proj_sem_label accessible via expected name
    self.sem_label_proj = self.proj_sem_label