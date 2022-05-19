from turtle import color
import open3d as o3d
import numpy as np
import cv2
import cv_bridge

from sensor_msgs.msg import Image, CameraInfo

bridge = cv_bridge.CvBridge()

def ros_to_rgbd(color_msg: Image, depth_msg: Image) -> o3d.geometry.RGBDImage:
    depth_img = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
    color_img = bridge.imgmsg_to_cv2(color_msg, desired_encoding='passthrough')

    depth_raw = o3d.geometry.Image(depth_img.astype(np.float32))
    color_raw = o3d.geometry.Image(color_img.astype(np.uint8))
    
    return o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)

def depth_msg_to_pcd(depth_msg: Image, camera_model: o3d.camera.PinholeCameraIntrinsic):
    depth_img = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

    cam_K = np.asarray(camera_model.intrinsic_matrix)
    
    fx = cam_K[0, 0]
    fy = cam_K[1, 1]
    cx = cam_K[0, 2]
    cy = cam_K[1, 2]

    [height, width] = depth_img.shape
    nx = np.linspace(0, width-1, width)
    ny = np.linspace(0, height-1, height)
    u, v = np.meshgrid(nx, ny)

    # Calculate numpy point cloud from depth data and params
    x = (u.flatten() - cx) / fx
    y = (v.flatten() - cy) / fy

    z = depth_img.flatten() / 1000
    x = np.multiply(x,z)
    y = np.multiply(y,z)

    xyz = np.zeros((np.size(x), 3))
    xyz[:, 0] = np.reshape(x, -1)
    xyz[:, 1] = np.reshape(y, -1)
    xyz[:, 2] = np.reshape(z, -1)

    # Create an open3d pointcloud object from realsense depth data
    pcd = o3d.geometry.PointCloud()
    
    mask = xyz[:,2] < 1
    pcd.points = o3d.utility.Vector3dVector(xyz[mask])
    pcd.colors = o3d.utility.Vector3dVector(xyz[mask])

    return pcd
    
        
def camera_info_to_camera_model(camera_info: CameraInfo) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(
      camera_info.width, camera_info.height,
      camera_info.K[0], camera_info.K[4], #fx, fy
      camera_info.K[2], camera_info.K[5] #cx, cy
    )

def rgbd_to_pcd_t(rgbd: o3d.geometry.RGBDImage, camera_model: o3d.camera.PinholeCameraIntrinsic) -> o3d.t.geometry.PointCloud:
    return o3d.t.geometry.PointCloud.create_from_rgbd_image(
      rgbd,
      camera_model,
    )

def rgbd_to_pcd(rgbd: o3d.geometry.RGBDImage, camera_model: o3d.camera.PinholeCameraIntrinsic) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud.create_from_rgbd_image(
      rgbd,
      camera_model,
    )

def pcd_to_mask(pcd: o3d.geometry.PointCloud, camera_model: o3d.camera.PinholeCameraIntrinsic):
  mask = np.zeros(shape=[camera_model.height, camera_model.width,3])
  uv = xyz_to_uv(pcd, camera_model)
  mask[uv[:,1], uv[:,0],:] = 255
  return mask

def xyz_to_uv(pcd: o3d.geometry.PointCloud, camera_model: o3d.camera.PinholeCameraIntrinsic):
    cam_K = np.asarray(camera_model.intrinsic_matrix)
    points = np.asarray(pcd.points)

    fx = cam_K[0, 0]
    fy = cam_K[1, 1]
    cx = cam_K[0, 2]
    cy = cam_K[1, 2]

    u = points[:,0] * fx / points[:,2] + cx
    v = points[:,1] * fy / points[:,2] + cy

    return np.c_[u.astype(np.int32), v.astype(np.int32)]


def reproject(pcd: o3d.geometry.PointCloud, camera_info: CameraInfo, img: np.ndarray = None):
  indices, _ = cv2.projectPoints(
    np.asarray(pcd.points), 
    np.identity(3), 
    np.zeros(shape=(1,3)), 
    np.array(camera_info.K).reshape((3,3)), 
    camera_info.D
  )

  indices = indices.astype(np.int32)

  if img is None:
    img = np.zeros(shape=(camera_info.height, camera_info.width, 3))
  
  colors = np.asarray(pcd.colors)
  
  for idx, p in enumerate(indices):
    x, y = p[0]
    cv2.circle(img, (x,y), 2, colors[idx] if colors.shape[0] != 0 else (1,0,1), -1)

  return img
