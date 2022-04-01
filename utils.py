import numpy as np
import open3d as o3d


def save_to_ply(verts, path):
    with open(path, 'w') as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write("element vertex %d\n" % verts.shape[0])
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write("end_header\n")
        for i in range(verts.shape[0]):
            fp.write('%f %f %f\n' % (verts[i, 0], verts[i, 1], verts[i, 2]))


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / (np.linalg.norm(axis_) + 1e-6)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length
            )
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            # cylinder_segment = cylinder_segment.translate(translation)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=translation
                )
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


def merge_meshes(meshes):
    # Compute total number of vertices and faces.
    num_vertices = 0
    num_triangles = 0
    num_vertex_colors = 0
    for i in range(len(meshes)):
        num_vertices += np.asarray(meshes[i].vertices).shape[0]
        num_triangles += np.asarray(meshes[i].triangles).shape[0]
        num_vertex_colors += np.asarray(meshes[i].vertex_colors).shape[0]

    # Merge vertices and faces.
    vertices = np.zeros((num_vertices, 3), dtype=np.float64)
    triangles = np.zeros((num_triangles, 3), dtype=np.int32)
    vertex_colors = np.zeros((num_vertex_colors, 3), dtype=np.float64)

    vertex_offset = 0
    triangle_offset = 0
    vertex_color_offset = 0
    for i in range(len(meshes)):
        current_vertices = np.asarray(meshes[i].vertices)
        current_triangles = np.asarray(meshes[i].triangles)
        current_vertex_colors = np.asarray(meshes[i].vertex_colors)

        vertices[vertex_offset:vertex_offset + current_vertices.shape[0]] = current_vertices
        triangles[triangle_offset:triangle_offset + current_triangles.shape[0]] = current_triangles + vertex_offset
        vertex_colors[vertex_color_offset:vertex_color_offset + current_vertex_colors.shape[0]] = current_vertex_colors

        vertex_offset += current_vertices.shape[0]
        triangle_offset += current_triangles.shape[0]
        vertex_color_offset += current_vertex_colors.shape[0]

    # Create a merged mesh object.
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color([1, 0, 0])
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh


def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi

    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.shape[1]):

        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1

        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_x, flow_y, rad_thresh=None, convert_to_bgr=False):
    if rad_thresh is not None:
        rad_max = rad_thresh
    else:
        rad = np.sqrt(np.square(flow_x) + np.square(flow_y))
        rad_max = np.max(rad)
        rad_max = min(rad_max, 100.0)

    epsilon = 1e-5
    flow_x = flow_x / (rad_max + epsilon)
    flow_y = flow_y / (rad_max + epsilon)

    return flow_compute_color(flow_x, flow_y, convert_to_bgr)


def get_graph_with_edge(samples, nn_idx):
    graph_nodes_mesh = []
    for node in samples:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([0.1, 0.1, 0.8])
        mesh_sphere.translate(node)
        graph_nodes_mesh.append(mesh_sphere)

    # Merge all different sphere meshes
    graph_nodes_mesh = merge_meshes(graph_nodes_mesh)
    
    # Graph edges
    edges_pairs = []
    for i in range(nn_idx.shape[0]):
        for j in range(nn_idx.shape[1]):
            if i != nn_idx[i, j]:
                edges_pairs.append([i, nn_idx[i, j]])

    colors = [[0.2, 0.2, 0.2] for i in range(len(edges_pairs))]
    line_mesh = LineMesh(samples, edges_pairs, colors, radius=0.001)
    line_mesh_geoms = line_mesh.cylinder_segments

    # Merge all different line meshes
    line_mesh_geoms = merge_meshes(line_mesh_geoms)

    # Combined nodes & edges
    rendered_graph = [graph_nodes_mesh, line_mesh_geoms]

    return merge_meshes(rendered_graph)


def get_colored_node(nodes, node_colors):
    graph_nodes_mesh = []
    for node, color in zip(nodes, node_colors):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([color[0], color[1], color[2]])
        mesh_sphere.translate(node)
        graph_nodes_mesh.append(mesh_sphere)

    # Merge all different sphere meshes
    graph_nodes_mesh = merge_meshes(graph_nodes_mesh)
    return graph_nodes_mesh


def render(geo, save_path):
    vis = o3d.visualization.Visualizer()
    parameters = o3d.io.read_pinhole_camera_parameters('./ScreenCamera.json')
    vis.create_window(width=480, height=480, left=0, top=0, visible=False)
    vis.add_geometry(geo)
    view_ctl = vis.get_view_control()
    view_ctl.convert_from_pinhole_camera_parameters(parameters)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)
    # del view_ctl


def rigid_icp(pc0, pc1):
    c0 = np.mean(pc0, axis=0)
    c1 = np.mean(pc1, axis=0)
    H = (pc0 - c0).transpose() @ (pc1 - c1)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t = c1 - R @ c0
    return R, t
