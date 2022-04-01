import os
import numpy as np
import open3d as o3d
import cv2
import imageio
import copy
import argparse
from tqdm import tqdm
from glob import glob
from utils import get_graph_with_edge, get_colored_node, merge_meshes, render, flow_to_color, save_to_ply


class Visualizer:
    def __init__(self, file_path):
        self.file_path = file_path
        # intrinsic matrix of the test sequences of the DeepDeform dataset
        self.intrinsic_mat = np.eye(3)
        self.intrinsic_mat[0, 2] = 323.172
        self.intrinsic_mat[1, 2] = 236.417
        self.intrinsic_mat[0, 0] = 575.548
        self.intrinsic_mat[1, 1] = 577.460
        self.cam_backward = 1.3
        self.cam_rotate_y = -np.pi / 4
        self.motion_scale_viz = 10.0

        self.input_path_node = os.path.join(self.file_path, 'input/node')
        self.input_path_graph = os.path.join(self.file_path, 'input/graph')

        self.total_frame = len(glob(self.input_path_node + '/*.npy'))

        self.output_path = os.path.join(self.file_path, 'output')
        self.output_path_node = os.path.join(self.file_path, 'output/node')
        self.output_path_complete = os.path.join(self.file_path, 'output/viz_complete')
        self.output_path_visible = os.path.join(self.file_path, 'output/viz_visible')
        self.output_path_history = os.path.join(self.file_path, 'output/viz_history')
        self.output_path_confidence = os.path.join(self.file_path, 'output/viz_confidence')
        self.output_path_graph = os.path.join(self.file_path, 'output/viz_graph')

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        if not os.path.exists(self.output_path_node):
            os.mkdir(self.output_path_node)
        if not os.path.exists(self.output_path_complete):
            os.mkdir(self.output_path_complete)
        if not os.path.exists(self.output_path_visible):
            os.mkdir(self.output_path_visible)
        if not os.path.exists(self.output_path_history):
            os.mkdir(self.output_path_history)
        if not os.path.exists(self.output_path_confidence):
            os.mkdir(self.output_path_confidence)
        if not os.path.exists(self.output_path_graph):
            os.mkdir(self.output_path_graph)

    def motion_project(self, pos, motion):
        # project 3d motion to 2d by intrinsic matrix
        uv = np.dot(pos, self.intrinsic_mat.T)
        uv = uv / uv[:, 2].reshape((-1, 1))
        uv = uv[:, :2]

        uv_moved = np.dot(pos + motion, self.intrinsic_mat.T)
        uv_moved = uv_moved / uv_moved[:, 2].reshape((-1, 1))
        uv_moved = uv_moved[:, :2]

        return uv_moved - uv

    def get_geo(self, frame_id):
        input = np.load(os.path.join(self.input_path_node, '{:04d}.npy'.format(frame_id)))
        graph = np.load(os.path.join(self.input_path_graph, '{:04d}.npz'.format(frame_id)))
        output = np.load(os.path.join(self.output_path_node, '{:04d}.npy'.format(frame_id)))

        node_pos = input[:, :3]
        visible_motion = input[:, 3:6]
        visible = input[:, -1] > 0.5
        complete_motion = output[:, :3]
        confidence = output[:, -1]
        nn_indices = graph['nn_index_l0']

        if frame_id > 1:
            input_prev = np.load(os.path.join(self.input_path_node, '{:04d}.npy'.format(frame_id - 1)))
            node_pos_prev = input_prev[:, :3]
            node_num_prev = node_pos_prev.shape[0]
            # node num of current frame could be larger than the previous frame, and new nodes will be add to the end of the node array
            historical_motion = node_pos[:node_num_prev] - node_pos_prev
        else:
            node_pos_prev = copy.deepcopy(node_pos)
            historical_motion = np.zeros_like(node_pos_prev)

        complete_motion_2d = self.motion_project(node_pos, complete_motion)
        complete_motion_color = flow_to_color(complete_motion_2d.reshape(-1, 1, 2)[:, :, 0], complete_motion_2d.reshape(-1, 1, 2)[:, :, 1], rad_thresh=self.motion_scale_viz, convert_to_bgr=False)
        complete_motion_color = complete_motion_color.reshape(-1, 3) / 255.0

        visible_motion_2d = self.motion_project(node_pos, visible_motion)
        visible_motion_color = flow_to_color(visible_motion_2d.reshape(-1, 1, 2)[:, :, 0], visible_motion_2d.reshape(-1, 1, 2)[:, :, 1], rad_thresh=self.motion_scale_viz, convert_to_bgr=False)
        visible_motion_color = visible_motion_color.reshape(-1, 3) / 255.0

        historical_motion_2d = self.motion_project(node_pos_prev, historical_motion)
        historical_motion_color = flow_to_color(historical_motion_2d.reshape(-1, 1, 2)[:, :, 0], historical_motion_2d.reshape(-1, 1, 2)[:, :, 1], rad_thresh=self.motion_scale_viz, convert_to_bgr=False)
        historical_motion_color = historical_motion_color.reshape(-1, 3) / 255.0

        confidence_color = (confidence * 0.7).reshape((-1, 1))
        confidence_color = np.tile(confidence_color, (1, 3))

        # # confidence in jet color
        # confidence_color = np.clip(confidence * 255, a_min=0, a_max=255)
        # confidence_color = confidence.astype(np.uint8)
        # confidence_color = cv2.applyColorMap(confidence_color, cv2.COLORMAP_JET)
        # confidence_color = confidence_color.reshape((-1, 3)) / 255.0

        # change the nodes position from camera to world
        node_pos[:, 1], node_pos[:, 2] = -node_pos[:, 1], -node_pos[:, 2] + self.cam_backward
        node_pos_prev[:, 1], node_pos_prev[:, 2] = -node_pos_prev[:, 1], -node_pos_prev[:, 2] + self.cam_backward

        node_meshs = get_colored_node(node_pos, complete_motion_color)
        complete_motion_mesh = merge_meshes([node_meshs])

        node_meshs = get_colored_node(node_pos[visible], visible_motion_color[visible])
        visible_motion_mesh = merge_meshes([node_meshs])

        node_meshs = get_colored_node(node_pos_prev, historical_motion_color)
        historical_motion_mesh = merge_meshes([node_meshs])

        node_meshs = get_colored_node(node_pos, confidence_color)
        confidence_mesh = merge_meshes([node_meshs])

        graph_mesh = get_graph_with_edge(node_pos, nn_indices)

        return complete_motion_mesh, visible_motion_mesh, historical_motion_mesh, confidence_mesh, graph_mesh

    def render(self, frame_id):
        complete_motion_mesh, visible_motion_mesh, historical_motion_mesh, confidence_mesh, graph_mesh = self.get_geo(frame_id)

        # rotate the objects to render in novel view
        complete_motion_mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, self.cam_rotate_y, 0])), np.zeros(3))
        visible_motion_mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, self.cam_rotate_y, 0])), np.zeros(3))
        historical_motion_mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, self.cam_rotate_y, 0])), np.zeros(3))
        confidence_mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, self.cam_rotate_y, 0])), np.zeros(3))
        graph_mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, self.cam_rotate_y, 0])), np.zeros(3))

        # render the objects to image
        render(complete_motion_mesh, os.path.join(self.output_path_complete, '{:04d}.png'.format(frame_id)))
        render(visible_motion_mesh, os.path.join(self.output_path_visible, '{:04d}.png'.format(frame_id)))
        render(historical_motion_mesh, os.path.join(self.output_path_history, '{:04d}.png'.format(frame_id)))
        render(confidence_mesh, os.path.join(self.output_path_confidence, '{:04d}.png'.format(frame_id)))
        render(graph_mesh, os.path.join(self.output_path_graph, '{:04d}.png'.format(frame_id)))

    def render_all(self):
        for frame_id in tqdm(range(1, self.total_frame + 1)):
            self.render(frame_id)

    def render_to_video(self):
        self.render_all()

        height = 480
        width = 480
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(self.output_path + '/out.avi', fourcc, 30.0, (width * 3, height * 3 + 60))

        motion_visible_files = glob(self.output_path_visible + '/*.png')
        motion_history_files = glob(self.output_path_history + '/*.png')
        graph_in_files = glob(self.output_path_graph + '/*.png')
        motion_out_files = glob(self.output_path_complete + '/*.png')
        confidence_out_files = glob(self.output_path_confidence + '/*.png')

        for frame_id in tqdm(range(self.total_frame)):
            result = np.ones([height * 3 + 60, width * 3, 3], np.uint8) * 255

            result[20:height + 20, 40:width + 40, :] = cv2.imread(motion_visible_files[frame_id])
            result[height + 20:2 * height + 20, 120:width + 120, :] = cv2.imread(motion_history_files[frame_id])
            result[2 * height + 20:3 * height + 20, 40:width + 40, :] = cv2.imread(graph_in_files[frame_id])

            result[240:height + 240, 2 * width - 60:3 * width - 60, :] = cv2.imread(motion_out_files[frame_id])
            result[height + 240:2 * height + 240, 2 * width - 60:3 * width - 60, :] = cv2.imread(confidence_out_files[frame_id])

            legend_bar = np.linspace(start=0.7, stop=0.0, num=100).reshape(-1, 1)
            legend_bar = (legend_bar * 255).astype(np.uint8)
            legend_bar = np.tile(legend_bar, 20)
            result[height + 460:height + 560, 2 * width + 360:2 * width + 380, :] = np.tile(legend_bar.reshape(100, 20, 1), (1, 1, 3))

            cv2.arrowedLine(result, (700, 720), (800, 720), (0, 0, 0), 2, cv2.LINE_AA, 0, 0.3)
            cv2.circle(result, (60, 720), 5, (0, 0, 0), 2)
            cv2.circle(result, (80, 720), 5, (0, 0, 0), 2)
            cv2.circle(result, (100, 720), 5, (0, 0, 0), 2)
            cv2.circle(result, (120, 720), 5, (0, 0, 0), 2)
            cv2.circle(result, (140, 720), 5, (0, 0, 0), 2)
            cv2.circle(result, (160, 720), 5, (0, 0, 0), 2)

            cv2.putText(result, "INPUT", (200, 30), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(result, "OUTPUT", (1080, 200), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

            cv2.putText(result, "visible motion", (140, 510), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.9, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(result, "historical motion sequence", (80, 990), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.9, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(result, "complete node graph", (140, 1470), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.9, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

            cv2.putText(result, "complete motion", (1000, 730), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.9, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(result, "confidence", (1080, 1210), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.9, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

            cv2.putText(result, "1", (1340, 960), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.9, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(result, "0", (1340, 1040), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.9, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

            out.write(result)
            cv2.imshow('result', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()

    def create_teaser(self):
        height = 480
        width = 480

        motion_visible_files = glob(self.output_path_visible + '/*.png')
        motion_history_files = glob(self.output_path_history + '/*.png')
        graph_in_files = glob(self.output_path_graph + '/*.png')
        motion_out_files = glob(self.output_path_complete + '/*.png')
        confidence_out_files = glob(self.output_path_confidence + '/*.png')

        with imageio.get_writer('./media/teaser.gif', mode='I') as writer:
            for frame_id in tqdm(range(700, 820)):
                result = np.ones([height * 3 + 60, width * 3, 3], np.uint8) * 255

                result[20:height + 20, 40:width + 40, :] = cv2.imread(motion_visible_files[frame_id])
                result[height + 20:2 * height + 20, 120:width + 120, :] = cv2.imread(motion_history_files[frame_id])
                result[2 * height + 20:3 * height + 20, 40:width + 40, :] = cv2.imread(graph_in_files[frame_id])

                result[240:height + 240, 2 * width - 60:3 * width - 60, :] = cv2.imread(motion_out_files[frame_id])
                result[height + 240:2 * height + 240, 2 * width - 60:3 * width - 60, :] = cv2.imread(confidence_out_files[frame_id])

                legend_bar = np.linspace(start=0.7, stop=0.0, num=100).reshape(-1, 1)
                legend_bar = (legend_bar * 255).astype(np.uint8)
                legend_bar = np.tile(legend_bar, 20)
                result[height + 460:height + 560, 2 * width + 360:2 * width + 380, :] = np.tile(legend_bar.reshape(100, 20, 1), (1, 1, 3))

                cv2.arrowedLine(result, (700, 720), (800, 720), (0, 0, 0), 2, cv2.LINE_AA, 0, 0.3)
                cv2.circle(result, (60, 720), 5, (0, 0, 0), 2)
                cv2.circle(result, (80, 720), 5, (0, 0, 0), 2)
                cv2.circle(result, (100, 720), 5, (0, 0, 0), 2)
                cv2.circle(result, (120, 720), 5, (0, 0, 0), 2)
                cv2.circle(result, (140, 720), 5, (0, 0, 0), 2)
                cv2.circle(result, (160, 720), 5, (0, 0, 0), 2)

                cv2.putText(result, "INPUT", (200, 30), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(result, "OUTPUT", (1080, 200), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

                cv2.putText(result, "visible motion", (140, 510), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.9, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(result, "historical motion sequence", (80, 990), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.9, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(result, "complete node graph", (140, 1470), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.9, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

                cv2.putText(result, "complete motion", (1000, 730), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.9, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(result, "confidence", (1080, 1210), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.9, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

                cv2.putText(result, "1", (1340, 960), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.9, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(result, "0", (1340, 1040), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.9, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

                writer.append_data(result)

    def view_in_3d(self, frame_id):
        complete_motion_mesh, visible_motion_mesh, historical_motion_mesh, confidence_mesh, graph_mesh = self.get_geo(frame_id)

        visible_motion_mesh.translate(np.array([-0.5, 0.8, 0]).reshape((3, 1)))
        graph_mesh.translate(np.array([-0.5, -0.8, 0]).reshape((3, 1)))
        complete_motion_mesh.translate(np.array([0.5, 0.8, 0]).reshape((3, 1)))
        confidence_mesh.translate(np.array([0.5, -0.8, 0]).reshape((3, 1)))

        o3d.visualization.draw_geometries([complete_motion_mesh, visible_motion_mesh, confidence_mesh, graph_mesh])

    def view_pyramid_3d(self, frame_id):
        input = np.load(os.path.join(self.file_path, 'input/node/{:04d}.npy'.format(frame_id)))
        graph = np.load(os.path.join(self.file_path, 'input/graph/{:04d}.npz'.format(frame_id)))

        node_pos = input[:, :3]

        # change the nodes position from camera to world
        node_pos[:, 1], node_pos[:, 2] = -node_pos[:, 1], -node_pos[:, 2] + self.cam_backward

        down_sample_idx1 = graph['down_sample_idx1']
        down_sample_idx2 = graph['down_sample_idx2']
        down_sample_idx3 = graph['down_sample_idx3']
        nn_index_l0 = graph['nn_index_l0']
        nn_index_l1 = graph['nn_index_l1']
        nn_index_l2 = graph['nn_index_l2']
        nn_index_l3 = graph['nn_index_l3']

        graph_mesh_l0 = get_graph_with_edge(node_pos, nn_index_l0)
        graph_mesh_l1 = get_graph_with_edge(node_pos[down_sample_idx1], nn_index_l1)
        graph_mesh_l2 = get_graph_with_edge(node_pos[down_sample_idx1][down_sample_idx2], nn_index_l2)
        graph_mesh_l3 = get_graph_with_edge(node_pos[down_sample_idx1][down_sample_idx2][down_sample_idx3], nn_index_l3)

        graph_mesh_l0.translate(np.array([-1.2, 0, 0]).reshape((3, 1)))
        graph_mesh_l1.translate(np.array([-0.4, 0, 0]).reshape((3, 1)))
        graph_mesh_l2.translate(np.array([0.4, 0, 0]).reshape((3, 1)))
        graph_mesh_l3.translate(np.array([1.2, 0, 0]).reshape((3, 1)))

        o3d.visualization.draw_geometries([graph_mesh_l0, graph_mesh_l1, graph_mesh_l2, graph_mesh_l3])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='render_to_video')
    parser.add_argument('--frame_id', default=790, type=int)
    args = parser.parse_args()

    visualizer = Visualizer(file_path='./data/')

    if args.mode == 'render':
        visualizer.render_all()
    elif args.mode == 'render_to_video':
        visualizer.render_to_video()
    elif args.mode == 'view3d':
        visualizer.view_in_3d(args.frame_id)
    elif args.mode == 'view_pyramid':
        visualizer.view_pyramid_3d(args.frame_id)


