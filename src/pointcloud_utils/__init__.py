import open3d as o3d
import numpy as np
import cv2
import cv_bridge
import tf2_ros
import rospy
from geometry_msgs.msg import Pose, TransformStamped
from sensor_msgs.msg import Image, CameraInfo

bridge = cv_bridge.CvBridge()

def ros_to_rgbd(color_msg: Image, depth_msg: Image, corner1=None, corner2=None) -> o3d.geometry.RGBDImage:
    depth_img = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
    color_img = bridge.imgmsg_to_cv2(color_msg, desired_encoding='passthrough')

    
    if corner1 is not None and corner2 is not None:
      mask = np.zeros(color_img.shape[:2], dtype=np.uint8)
      mask = cv2.rectangle(mask, corner1, corner2, 255, -1)
      # cv2.imshow('Mask', mask)
      color_img = cv2.bitwise_and(color_img, color_img, mask=mask)
      # masked_depth = cv2.bitwise_and(depth_img, depth_img, mask=mask) * 255
        
      # cv2.imshow('Depth', masked_depth)
      # cv2.imshow('Color', color_img)
      # cv2.waitKey(0)
    depth_raw = o3d.geometry.Image(depth_img.astype(np.float32))
    color_raw = o3d.geometry.Image(color_img.astype(np.uint8))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
        
def camera_info_to_camera_model(camera_info: CameraInfo) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(
      camera_info.width, camera_info.height,
      camera_info.K[0], camera_info.K[4], #fx, fy
      camera_info.K[2], camera_info.K[5] #cx, cy
    )

def rgbd_to_pcd_t(rgbd: o3d.geometry.RGBDImage, camera_model: o3d.camera.PinholeCameraIntrinsic) -> o3d.t.geometry.PointCloud:
    print(camera_model)
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

##### ------------------------ Image Helper Class --------------------------------------------
class ImageHelper(object):
    def __init__(self):
        pass

    def xyz_to_pixel(self, x, y, z, camera_model: o3d.camera.PinholeCameraIntrinsic):
        """
        Converts a xyz (m) coord to 2D pixel (u,v) using camera_model
        :param x: x coordinate (m)
        :type x: float32
        :param y: y coordinate (m)
        :type y: float32
        :param z: z coordinate (m)
        :type x: float32
        :return pixel: pixel coordinate transformed
        :rtype pixel: tuple of two elements
        """
        cam_K = np.asarray(camera_model.intrinsic_matrix)
        fx = cam_K[0, 0]
        fy = cam_K[1, 1]
        cx = cam_K[0, 2]
        cy = cam_K[1, 2]

        if not x:
            x = 0.0
        if not y:
            y = 0.0
        if not z:
            z = 0.0

        height = z
        if height:
            u = ((x * fx) / height) + cx
            v = ((y * fy) / height) + cy
            return (u,v)
        else:
            return (None,None)

    def pixel_to_xy(self, u, v, depth, camera_model: o3d.camera.PinholeCameraIntrinsic):
        """
        Converts a 2D pixel (u,v) to xy (m) using camera_model and a defined depth (m)
        :param u: x coordinate
        :type u: int32
        :param v: y coordinate
        :type v: int32
        :param depth: depth from camera to compute x and y
        :type depth: float32
        :return xyz: pixel coordinate transformed
        :rtype xyz: tuple of two elements
        """
        cam_K = np.asarray(camera_model.intrinsic_matrix)
        fx = cam_K[0, 0]
        fy = cam_K[1, 1]
        cx = cam_K[0, 2]
        cy = cam_K[1, 2]

        if not u:
            u = 0
        if not v:
            v = 0
        if not depth:
            depth = 0.0

        if fx and fy:
            x = ((u - cx) * depth) / fx
            y = ((v - cy) * depth) / fy
            return (x,y)
        else:
            return (None,None)

##### ------------------------ ROS Helper Class --------------------------------------------
class ROSHelper(object):
    def __init__(self) -> None:
        pass

    def tf_to_pose(self, base, target, tf_buffer) -> Pose():
        """
        Gets a transform and converts to a Pose object
        """
        pose = Pose()
        try:
            # Try to get transform frame from tf2
            trans = tf_buffer.lookup_transform(
                base,
                target,
                rospy.Time()
            )

            pose.position.x = trans.transform.translation.x
            pose.position.y = trans.transform.translation.y
            pose.position.z = trans.transform.translation.z
            pose.orientation.x = trans.transform.rotation.x
            pose.orientation.y = trans.transform.rotation.y
            pose.orientation.z = trans.transform.rotation.z
            pose.orientation.w = trans.transform.rotation.w
        except(tf2_ros.LookupException, 
            tf2_ros.ConnectivityException, 
            tf2_ros.ExtrapolationException):
            pass

        return pose
