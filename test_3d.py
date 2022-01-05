import open3d as o3d
import numpy as np


def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=800, height=600)
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=800, height=600)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    file_path = '/home/chriskafka/dataset/ShapeNetCompletion/train/complete/02933112/1a1b62a38b2584874c62bee40dcdc539.pcd'
    pcd = o3d.io.read_point_cloud(file_path)
    save_view_point(pcd, "viewpoint.json")
    load_view_point(pcd, "viewpoint.json")
