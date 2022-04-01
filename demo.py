import os
import torch
from glob import glob
import numpy as np
from tqdm import tqdm
from utils import rigid_icp
from model import MotionCompleteNet


class Demo:
    def __init__(self, model, input_path, output_path):
        self.model = model
        self.input_path = input_path
        self.input_path_node = os.path.join(self.input_path, 'node')
        self.input_path_graph = os.path.join(self.input_path, 'graph')
        self.output_path = output_path
        self.output_path_node = os.path.join(self.output_path, 'node')
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        if not os.path.exists(self.output_path_node):
            os.mkdir(self.output_path_node)
        self.historical_motion = None
        self.historical_max_len = 16
        self.std_curr = None
        self.std_prev = None
        self.rigid_motion_curr = None

    def preprocess(self, frame_id):
        node_feature = np.load(os.path.join(self.input_path_node, '{:04d}.npy'.format(frame_id)))
        node_pos = node_feature[:, :3]
        node_motion = node_feature[:, 3:6]
        visible = node_feature[:, -1] > 0.5

        pyd = np.load(os.path.join(self.input_path_graph, '{:04d}.npz'.format(frame_id)))
        down_sample_idx1 = pyd['down_sample_idx1']
        down_sample_idx2 = pyd['down_sample_idx2']
        down_sample_idx3 = pyd['down_sample_idx3']
        up_sample_idx1 = pyd['up_sample_idx1']
        up_sample_idx2 = pyd['up_sample_idx2']
        up_sample_idx3 = pyd['up_sample_idx3']
        nn_index_l0 = pyd['nn_index_l0']
        nn_index_l1 = pyd['nn_index_l1']
        nn_index_l2 = pyd['nn_index_l2']
        nn_index_l3 = pyd['nn_index_l3']

        node_num_l0 = node_pos.shape[0]

        # extract rigid motion
        rigid_R, rigid_t = rigid_icp(node_pos[visible, :], node_pos[visible, :] + node_motion[visible, :])
        self.rigid_motion_curr = np.dot(node_pos, rigid_R.transpose()) + rigid_t - node_pos
        nonrigid_motion = node_motion - self.rigid_motion_curr

        curr_motion = np.zeros(shape=(node_num_l0, 4))
        # motion in centimeter
        curr_motion[visible, :3] = nonrigid_motion[visible, :] * 100.0

        # normalize the motion
        self.curr_std = np.mean(np.std(curr_motion[visible, :3], axis=0)) + 0.1
        curr_motion[visible, :3] = curr_motion[visible, :3] / self.curr_std
        curr_motion[:, -1] = visible

        # init the mu of new nodes as 0.0, and the sigma of new nodes as a larger value (1.0)
        prev_motion = np.zeros(shape=(node_num_l0, 4))
        prev_motion[:, -1] = 1.0

        # for the first frame, set historical motion
        # using node position change between consequent frames as historical motion
        if frame_id > 1:
            node_feature_prev = np.load(os.path.join(self.input_path_node, '{:04d}.npy'.format(frame_id - 1)))
            node_pos_prev = node_feature_prev[:, :3]
            visible_prev = node_feature_prev[:, -1] > 0.5
            prev_node_num = node_pos_prev.shape[0]

            # node num of current frame could be larger than the previous frame, and new nodes will be add to the end of the node array
            node_motion_prev = node_pos[:node_pos_prev.shape[0]] - node_pos_prev

            rigid_R, rigid_t = rigid_icp(node_pos_prev[visible_prev, :], node_pos_prev[visible_prev, :] + node_motion_prev[visible_prev, :])
            rigid_motion_prev = np.dot(node_pos_prev, rigid_R.transpose()) + rigid_t - node_pos_prev
            prev_motion[:prev_node_num, :3] = (node_motion_prev - rigid_motion_prev) * 100.0

        if self.historical_motion is None:
            self.historical_motion = np.zeros(shape=(1, node_num_l0, 4))
        else:
            seq_len = self.historical_motion.shape[0]
            prev_node_num = self.historical_motion.shape[1]
            drop = (seq_len == self.historical_max_len) * 1
            seq_len = min(seq_len + 1, self.historical_max_len)
            temp = np.zeros(shape=(seq_len, node_num_l0, 4))
            temp[:-1, :prev_node_num, :] = self.historical_motion[drop:, :, :] * self.std_prev / self.curr_std
            temp[-1, :prev_node_num, :] = prev_motion[:prev_node_num, :] / self.curr_std
            self.historical_motion = temp

        self.std_prev = self.curr_std

        node_pos = node_pos - np.mean(node_pos, axis=0)

        node_pos_torch = torch.from_numpy(node_pos.astype(np.float32)).to(device)
        curr_motion_torch = torch.from_numpy(curr_motion.astype(np.float32)).to(device)
        historical_motion_torch = torch.from_numpy(self.historical_motion.astype(np.float32)).to(device)

        node_num, nn_num = nn_index_l0.shape
        edge_index_l0 = np.zeros(shape=(2, node_num * nn_num), dtype=np.int64)
        edge_index_l0[0:] = np.repeat(np.arange(node_num), nn_num)
        edge_index_l0[1:] = nn_index_l0.reshape(-1)

        node_num, nn_num = nn_index_l1.shape
        edge_index_l1 = np.zeros(shape=(2, node_num * nn_num), dtype=np.int64)
        edge_index_l1[0:] = np.repeat(np.arange(node_num), nn_num)
        edge_index_l1[1:] = nn_index_l1.reshape(-1)

        node_num, nn_num = nn_index_l2.shape
        edge_index_l2 = np.zeros(shape=(2, node_num * nn_num), dtype=np.int64)
        edge_index_l2[0:] = np.repeat(np.arange(node_num), nn_num)
        edge_index_l2[1:] = nn_index_l2.reshape(-1)

        node_num, nn_num = nn_index_l3.shape
        edge_index_l3 = np.zeros(shape=(2, node_num * nn_num), dtype=np.int64)
        edge_index_l3[0:] = np.repeat(np.arange(node_num), nn_num)
        edge_index_l3[1:] = nn_index_l3.reshape(-1)

        edge_index_l0 = torch.from_numpy(edge_index_l0).to(device)
        edge_index_l1 = torch.from_numpy(edge_index_l1).to(device)
        edge_index_l2 = torch.from_numpy(edge_index_l2).to(device)
        edge_index_l3 = torch.from_numpy(edge_index_l3).to(device)
        down_sample_idx1 = torch.from_numpy(np.array(down_sample_idx1).astype(np.int64)).to(device)
        down_sample_idx2 = torch.from_numpy(np.array(down_sample_idx2).astype(np.int64)).to(device)
        down_sample_idx3 = torch.from_numpy(np.array(down_sample_idx3).astype(np.int64)).to(device)
        up_sample_idx1 = torch.from_numpy(np.array(up_sample_idx1).astype(np.int64)).to(device)
        up_sample_idx2 = torch.from_numpy(np.array(up_sample_idx2).astype(np.int64)).to(device)
        up_sample_idx3 = torch.from_numpy(np.array(up_sample_idx3).astype(np.int64)).to(device)

        return node_pos_torch, curr_motion_torch, historical_motion_torch, \
               [edge_index_l0, edge_index_l1, edge_index_l2, edge_index_l3], \
               [down_sample_idx1, down_sample_idx2, down_sample_idx3], \
               [up_sample_idx1, up_sample_idx2, up_sample_idx3]

    def run_single_frame(self, frame_id):
        node_pos, curr_motion, historical_motion, edge_indices, down_sample_indices, up_sample_indices = self.preprocess(frame_id)

        outputs = self.model(node_pos, curr_motion, historical_motion, edge_indices,down_sample_indices, up_sample_indices)
        outputs = outputs.detach().cpu().numpy()
        mu = outputs[:, :3]
        sigma = outputs[:, -1]

        # eq.7 in the paper
        motion_scale = np.sqrt(np.sum(np.square(mu), axis=1))
        confidence = np.exp(-4 * np.square(sigma / (motion_scale + 1.0)))

        mu = mu * self.curr_std
        sigma = sigma * self.curr_std

        pred_motion = mu / 100.0
        node_motion = pred_motion + self.rigid_motion_curr

        return node_motion, confidence

    def run_demo(self):
        total_frame = len(glob(self.input_path_node + '/*.npy'))
        for frame_id in tqdm(range(1, total_frame + 1)):
            motion, confidence = self.run_single_frame(frame_id)
            np.save(os.path.join(self.output_path_node, '{:04d}.npy'.format(frame_id)), np.hstack((motion, confidence.reshape((-1, 1)))))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_path = './data/input/'
    output_path = './data/output/'
    checkpoint_path = './checkpoints/model_noise_all.tar'

    model = MotionCompleteNet().to(device)
    torch_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(torch_checkpoint['model_state_dict'])

    # load network input from files and save the predicted complete motion with confidence
    demo = Demo(model, input_path, output_path)
    demo.run_demo()


