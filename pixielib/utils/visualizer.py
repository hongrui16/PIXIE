


import open3d as o3d
import numpy as np
import os
import plotly.graph_objects as go
import keyboard


class vis_mesh_points():
    def __init__(self, height=1000, width=1000, face_filepath = None):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=width, height=height)
        self.vis.get_render_option().light_on = True
        if face_filepath is None:
            faces_filepath = r'C:\Users\hongr\Documents\GMU_research\computerVersion\hand_modeling\PIXIE\data\SMPLX_NEUTRAL_2020.npz'
        all_data = np.load(faces_filepath, allow_pickle=True)
        self.faces = all_data['f']
        self.pcd = o3d.geometry.PointCloud()

        


    def vis_mesh(self, vertices, output_dir = None, name = None, hold_vis = True):
        mesh = o3d.geometry.TriangleMesh()

        # 设置网格的顶点
        mesh.vertices = o3d.utility.Vector3dVector(vertices)

        # 设置网格的三角形面
        mesh.triangles = o3d.utility.Vector3iVector(self.faces)

        # 计算顶点的法线
        mesh.compute_vertex_normals()

        # 重置视图
        self.vis.clear_geometries()
        self.vis.add_geometry(mesh)

        # 光照和相机视角设置
        ctr = self.vis.get_view_control()
        ctr.set_front([0, 0, 1])
        ctr.set_lookat([0, -0.65, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.6)

        transformation = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, np.pi))
        mesh.rotate(transformation, center=mesh.get_center())

        self.vis.update_geometry(mesh)
        self.vis.poll_events()
        self.vis.update_renderer()


        if not output_dir is None and not name is None:
            os.makedirs(output_dir, exist_ok=True)
            # 保存当前视角下的图像
            output_filepath = os.path.join(output_dir, f'{name}_mesh.jpg')
            print(f'save image to {output_filepath}')
            self.vis.capture_screen_image(output_filepath)


    def vis_points(self, points, i, output_dir = None, name = None, hold_vis = True):
        self.pcd.points = o3d.utility.Vector3dVector(points)
        
        # 如果是第一次迭代，需要添加点云到可视化窗口
        if i == 0:
            self.vis.add_geometry(self.pcd)
        
        # while True:
            # 更新点云
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        while hold_vis:
            self.vis.poll_events()
            self.vis.update_renderer()

            if keyboard.is_pressed('esc'):  # 检查是否按下了ESC键
                # print("ESC pressed, exiting...")
                break  # 退出循环

            #     # 捕获ESC按键
            #     if self.vis.get_view_control().get_interactive_status().escaped:
            #         break
            # else:
            #     break


        if not output_dir is None and not name is None:
            os.makedirs(output_dir, exist_ok=True)
            # 保存当前视角下的图像
            output_filepath = os.path.join(output_dir, f'{name}_points.jpg')
            self.vis.capture_screen_image(output_filepath)
        

    def destroy(self):
        self.vis.destroy_window()



def plotly_save_point_cloud(points, file_path='plotly_3d_plot.html'):
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=points[:, 2],  # color points by Z value
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )])

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.write_html(file_path)  # Save as HTML
