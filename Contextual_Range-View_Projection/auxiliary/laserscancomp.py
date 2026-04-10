#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from auxiliary.vispy_manager import VispyManager
import numpy as np
import os

from vispy.scene import Text


class LaserScanComp(VispyManager):
  """Class that creates and handles a side-by-side pointcloud comparison"""

  def __init__(self, scans, scan_names, label_names, offset=0, images=True, instances=False, link=False):
    super().__init__(offset, len(scan_names), images, instances)
    self.scan_a_view = None
    self.scan_a_vis = None
    self.scan_b_view = None
    self.scan_b_vis = None
    # ------------- new added
    self.scan_diff_view = None
    self.scan_diff_vis = None
    # -------------- end added
    self.inst_a_view = None
    self.inst_a_vis = None
    self.inst_b_view = None
    self.inst_b_vis = None
    self.img_a_view = None
    self.img_a_vis = None
    self.img_b_view = None
    self.img_b_vis = None
    self.img_diff_view = None
    self.img_diff_vis = None
    self.img_inst_a_view = None
    self.img_inst_a_vis = None
    self.img_inst_b_view = None
    self.img_inst_b_vis = None
    self.scan_a, self.scan_b = scans
    self.scan_names = scan_names
    self.label_a_names, self.label_b_names = label_names
    self.link = link

    self.reset()
    self.update_scan()

  def reset(self):
    """prepares the canvas(es) for the visualizer"""
    self.scan_a_view, self.scan_a_vis = super().add_viewbox(0, 0)
    self.scan_b_view, self.scan_b_vis = super().add_viewbox(0, 1)
    # ----------------- added new
    self.scan_diff_view, self.scan_diff_vis = super().add_viewbox(0, 2)
    # ------------------ end new


    # if self.link:
    #   self.scan_a_view.camera.link(self.scan_b_view.camera)
    if self.link:
        self.scan_a_view.camera.link(self.scan_b_view.camera)
        self.scan_a_view.camera.link(self.scan_diff_view.camera)


    if self.images:
      self.img_a_view, self.img_a_vis = super().add_image_viewbox(0, 0)
      self.img_b_view, self.img_b_vis = super().add_image_viewbox(1, 0)
      self.img_diff_view, self.img_diff_vis = super().add_image_viewbox(2, 0)

      if self.instances:
        self.img_inst_a_view, self.img_inst_a_vis = super().add_image_viewbox(3, 0)
        self.img_inst_b_view, self.img_inst_b_vis = super().add_image_viewbox(4, 0)

    if self.instances:
      self.inst_a_view, self.inst_a_vis = super().add_viewbox(1, 0)
      self.inst_b_view, self.inst_b_vis = super().add_viewbox(1, 1)

      if self.link:
        self.scan_a_view.camera.link(self.inst_a_view.camera)
        self.inst_a_view.camera.link(self.inst_b_view.camera)

  # def update_scan(self):
  #   """updates the scans, images and instances"""
  #   self.scan_a.open_scan(self.scan_names[self.offset])
  #   self.scan_a.open_label(self.label_a_names[self.offset])
  #   self.scan_a.colorize()
  #   self.scan_a_vis.set_data(self.scan_a.points,
  #                         face_color=self.scan_a.sem_label_color[..., ::-1],
  #                         edge_color=self.scan_a.sem_label_color[..., ::-1],
  #                         size=1)

  #   self.scan_b.open_scan(self.scan_names[self.offset])
  #   self.scan_b.open_label(self.label_b_names[self.offset])
  #   self.scan_b.colorize()
  #   self.scan_b_vis.set_data(self.scan_b.points,
  #                         face_color=self.scan_b.sem_label_color[..., ::-1],
  #                         edge_color=self.scan_b.sem_label_color[..., ::-1],
  #                         size=1)

  #   if self.instances:
  #     self.inst_a_vis.set_data(self.scan_a.points,
  #                              face_color=self.scan_a.inst_label_color[..., ::-1],
  #                              edge_color=self.scan_a.inst_label_color[..., ::-1],
  #                              size=1)
  #     self.inst_b_vis.set_data(self.scan_b.points,
  #                              face_color=self.scan_b.inst_label_color[..., ::-1],
  #                              edge_color=self.scan_b.inst_label_color[..., ::-1],
  #                              size=1)

  #   if self.images:
  #     self.img_a_vis.set_data(self.scan_a.proj_sem_color[..., ::-1])
  #     self.img_a_vis.update()
  #     self.img_b_vis.set_data(self.scan_b.proj_sem_color[..., ::-1])
  #     self.img_b_vis.update()

  #     # Add difference image: red = mismatch, white = match
  #     diff_mask = (self.scan_a.sem_label_proj != self.scan_b.sem_label_proj)
  #     diff_img = np.zeros((*diff_mask.shape, 3), dtype=np.uint8)
  #     diff_img[diff_mask] = [255, 0, 0]       # Red for mismatch
  #     diff_img[~diff_mask] = [255, 255, 255]  # White for match
  #     self.img_diff_vis.set_data(diff_img[..., ::-1])
  #     self.img_diff_vis.update()

  #     if self.instances:
  #       self.img_inst_a_vis.set_data(self.scan_a.proj_inst_color[..., ::-1])
  #       self.img_inst_a_vis.update()
  #       self.img_inst_b_vis.set_data(self.scan_b.proj_inst_color[..., ::-1])
  #       self.img_inst_b_vis.update()

  # def update_scan(self):
  #   """updates the scans, images and instances"""
  #   # Load and colorize scan A
  #   self.scan_a.open_scan(self.scan_names[self.offset])
  #   self.scan_a.open_label(self.label_a_names[self.offset])
  #   self.scan_a.colorize()
  #   self.scan_a_vis.set_data(self.scan_a.points,
  #                            face_color=self.scan_a.sem_label_color[..., ::-1],
  #                            edge_color=self.scan_a.sem_label_color[..., ::-1],
  #                            size=1)

  #   # Load and colorize scan B
  #   self.scan_b.open_scan(self.scan_names[self.offset])
  #   self.scan_b.open_label(self.label_b_names[self.offset])
  #   self.scan_b.colorize()
  #   self.scan_b_vis.set_data(self.scan_b.points,
  #                            face_color=self.scan_b.sem_label_color[..., ::-1],
  #                            edge_color=self.scan_b.sem_label_color[..., ::-1],
  #                            size=1)

  #   # Add 3D instance views (if enabled)
  #   if self.instances:
  #       self.inst_a_vis.set_data(self.scan_a.points,
  #                                face_color=self.scan_a.inst_label_color[..., ::-1],
  #                                edge_color=self.scan_a.inst_label_color[..., ::-1],
  #                                size=1)
  #       self.inst_b_vis.set_data(self.scan_b.points,
  #                                face_color=self.scan_b.inst_label_color[..., ::-1],
  #                                edge_color=self.scan_b.inst_label_color[..., ::-1],
  #                                size=1)

  #   # Add 2D projected image views
  #   if self.images:
  #       self.img_a_vis.set_data(self.scan_a.proj_sem_color[..., ::-1])
  #       self.img_a_vis.update()
  #       self.img_b_vis.set_data(self.scan_b.proj_sem_color[..., ::-1])
  #       self.img_b_vis.update()

  #       # Compute and show label difference (image)
  #       diff_mask = (self.scan_a.sem_label_proj != self.scan_b.sem_label_proj)
  #       diff_img = np.zeros((*diff_mask.shape, 3), dtype=np.uint8)
  #       diff_img[diff_mask] = [0, 0, 255]       # Red for mismatch
  #       diff_img[~diff_mask] = [255, 255, 255]  # White for match
  #       self.img_diff_vis.set_data(diff_img[..., ::-1])
  #       self.img_diff_vis.update()

  #       if self.instances:
  #           self.img_inst_a_vis.set_data(self.scan_a.proj_inst_color[..., ::-1])
  #           self.img_inst_a_vis.update()
  #           self.img_inst_b_vis.set_data(self.scan_b.proj_inst_color[..., ::-1])
  #           self.img_inst_b_vis.update()

  #   # Add 3D label difference view
  #   a_labels = self.scan_a.sem_label
  #   b_labels = self.scan_b.sem_label
  #   assert a_labels.shape == b_labels.shape

  #   agree_mask = a_labels == b_labels
  #   diff_color = np.ones((a_labels.shape[0], 3), dtype=np.float32)
  #   diff_color[~agree_mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Red for mismatches
  #   diff_color[agree_mask] = np.array([1.0, 1.0, 1.0], dtype=np.float32)   # White for matches

  #   self.scan_diff_vis.set_data(self.scan_a.points,
  #                               face_color=diff_color,
  #                               edge_color=diff_color,
  #                               size=1)


  def update_scan(self):
    """updates the scans, images and instances"""

    # === Auto-load center/weight files based on label filenames ===
    label_a_base = os.path.splitext(self.label_a_names[self.offset])[0]
    center_a_path = label_a_base + '.center.npy'
    weight_a_path = label_a_base + '.weight.npy'

    label_b_base = os.path.splitext(self.label_b_names[self.offset])[0]
    center_b_path = label_b_base + '.center.npy'
    weight_b_path = label_b_base + '.weight.npy'

    # Load and colorize scan A
    self.scan_a.open_scan(self.scan_names[self.offset],
                          center_file=center_a_path,
                          weight_file=weight_a_path)
    self.scan_a.open_label(self.label_a_names[self.offset])
    self.scan_a.colorize()
    self.scan_a_vis.set_data(self.scan_a.points,
                             face_color=self.scan_a.sem_label_color[..., ::-1],
                             edge_color=self.scan_a.sem_label_color[..., ::-1],
                             size=1)

    # Load and colorize scan B
    self.scan_b.open_scan(self.scan_names[self.offset],
                          center_file=center_b_path,
                          weight_file=weight_b_path)
    self.scan_b.open_label(self.label_b_names[self.offset])
    self.scan_b.colorize()
    self.scan_b_vis.set_data(self.scan_b.points,
                             face_color=self.scan_b.sem_label_color[..., ::-1],
                             edge_color=self.scan_b.sem_label_color[..., ::-1],
                             size=1)

    # Add 3D instance views (if enabled)
    if self.instances:
        self.inst_a_vis.set_data(self.scan_a.points,
                                 face_color=self.scan_a.inst_label_color[..., ::-1],
                                 edge_color=self.scan_a.inst_label_color[..., ::-1],
                                 size=1)
        self.inst_b_vis.set_data(self.scan_b.points,
                                 face_color=self.scan_b.inst_label_color[..., ::-1],
                                 edge_color=self.scan_b.inst_label_color[..., ::-1],
                                 size=1)

    # Add 2D projected image views
    if self.images:
        self.img_a_vis.set_data(self.scan_a.proj_sem_color[..., ::-1])
        self.img_a_vis.update()
        self.img_b_vis.set_data(self.scan_b.proj_sem_color[..., ::-1])
        self.img_b_vis.update()

        # Compute and show label difference (image)
        diff_mask = (self.scan_a.sem_label_proj != self.scan_b.sem_label_proj)
        diff_img = np.zeros((*diff_mask.shape, 3), dtype=np.uint8)
        diff_img[diff_mask] = [0, 0, 255]       # Red for mismatch
        diff_img[~diff_mask] = [255, 255, 255]  # White for match
        self.img_diff_vis.set_data(diff_img)  # Removed ::-1 if you're using RGB natively
        self.img_diff_vis.update()

        if self.instances:
            self.img_inst_a_vis.set_data(self.scan_a.proj_inst_color[..., ::-1])
            self.img_inst_a_vis.update()
            self.img_inst_b_vis.set_data(self.scan_b.proj_inst_color[..., ::-1])
            self.img_inst_b_vis.update()

    # Add 3D label difference view
    a_labels = self.scan_a.sem_label
    b_labels = self.scan_b.sem_label
    assert a_labels.shape == b_labels.shape

    agree_mask = a_labels == b_labels
    diff_color = np.ones((a_labels.shape[0], 3), dtype=np.float32)
    diff_color[~agree_mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Red for mismatches
    diff_color[agree_mask] = np.array([1.0, 1.0, 1.0], dtype=np.float32)   # White for matches

    self.scan_diff_vis.set_data(self.scan_a.points,
                                face_color=diff_color,
                                edge_color=diff_color,
                                size=1)
    
    # Get scan ID and sequence from path
    full_path = self.scan_names[self.offset]
    scan_file = os.path.basename(full_path)        # '001235.bin'
    scan_id = os.path.splitext(scan_file)[0]       # '001235'
    sequence = full_path.split(os.sep)[-3]         # '00'

    # Set dynamic window titles
    self.canvas.title = f"scan — Sequence {sequence} | Scan {scan_id}"
    if self.images:
        self.img_canvas.title = f"img — Sequence {sequence} | Scan {scan_id}"

