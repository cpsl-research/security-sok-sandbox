import os

from avapi.nuscenes import nuScenesManager
from avstack.modules.perception.object2dfv import MMDetObjectDetector2D
from avstack.modules.perception.object3d import MMDetObjectDetector3D
from avstack.modules.tracking.tracker3d import Ab3dmotTracker
from avstack.modules.prediction import KinematicPrediction
from avstack.geometry import GlobalOrigin3D


dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "..", "data")


if __name__ == "__main__":
    # -- load perception models
    try:
        M2D = MMDetObjectDetector2D(dataset="cityscapes", model="fasterrcnn")
        M3D = MMDetObjectDetector3D(dataset="nuscenes", model="pointpillars")
        tracker = Ab3dmotTracker()
        # ? Do not know how prediction works.
        # predictor = KinematicPrediction()
    except Exception as e:
        print("\nLoading perception models FAILED!")
        raise e
    else:
        print("\nLoading perception models SUCCEEDED\n")

    # -- perform inference
    try:
        data_path = os.path.join(data_dir, "nuScenes")
        SM = nuScenesManager(data_path)
        DM = SM.get_scene_dataset_by_index(0)

        ego = DM.get_ego(DM.frames[0])
        print(ego)

        # -- image inference
        img = DM.get_image(DM.frames[0], sensor="main_camera")
        dets_2d = M2D(img)
        print(dets_2d)

        # -- lidar inference
        pc = DM.get_lidar(DM.frames[0], sensor="lidar")
        dets_3d = M3D(pc)
        print(dets_3d)

        tracks = tracker(detections=dets_3d, platform=GlobalOrigin3D)
        print(tracks)

        # predictions = predictor(tracks, DM.frames[0])

    except Exception as e:
        print("\nRunning perception inference FAILED!")
        raise e
    else:
        print("\nRunning perception inference SUCCEEDED\n")
