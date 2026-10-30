import copy
import unittest

import cv2
import numpy as np
import scene_config as C
from cube_pose import silhouette
from test_support import camera_model
from check_pick_position import check_target


class PickPositionTests(unittest.TestCase):
    def setUp(self):
        self.cam = camera_model(C.TABLE_CAM)
        self.cam.j["camera"] = "table"
        self.xy = np.array([.35, .10])
        self.pick = dict(xy=self.xy.tolist(), colour="blue", detection_id=0,
                         pose_confidence=.95, eligible=True)
        self.bins = {"blue": dict(xy=[.315, .33], outer=[.08, .08])}

    def image(self, positions):
        image = np.full((self.cam.H, self.cam.W, 3), 180, np.uint8)
        for xy in positions:
            image[silhouette(self.cam, xy, 17, image.shape[:2]) > 0] = [190, 75, 30]
        return image

    def check(self, positions):
        return check_target(self.image(positions), self.cam.j, self.pick, self.bins)

    def test_unchanged_target(self):
        result = self.check([self.xy])
        self.assertEqual(result["status"], "position_consistent")
        self.assertLess(result["displacement_mm"], 1)
        self.assertFalse(result["plan_updated"])

    def test_moved_target(self):
        result = self.check([self.xy + [.010, 0]])
        self.assertEqual(result["status"], "target_moved")
        self.assertAlmostEqual(result["displacement_mm"], 10, delta=1)

    def test_uncertainty_band(self):
        self.assertEqual(self.check([self.xy + [.0035, 0]])["status"], "uncertain")

    def test_missing_or_outside_search_target(self):
        for positions in ([], [self.xy + [.05, 0]]):
            self.assertEqual(self.check(positions)["status"], "uncertain")

    def test_ambiguous_same_colour_targets(self):
        result = self.check([self.xy + [-.010, 0], self.xy + [.010, 0]])
        self.assertEqual(result["status"], "uncertain")
        self.assertGreater(result["candidate_count"], 1)

    def test_bin_objects_do_not_become_loose_target(self):
        self.bins = {"blue": dict(xy=self.xy.tolist(), outer=[.08, .08])}
        self.assertEqual(self.check([self.xy])["status"], "uncertain")

    def test_rejects_invalid_calibration_and_observation(self):
        image = self.image([self.xy])
        for key, value in (("resolution", [3840, 2160]), ("focal_length_mm", float("nan")),
                           ("camera", "side"), ("camera_to_world", [[0]*4]*4)):
            record = copy.deepcopy(self.cam.j)
            record[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_target(image, record, self.pick, self.bins)

    def test_rejects_unaccepted_plan_target(self):
        for key, value in (("pose_confidence", .3), ("eligible", False), ("xy", [float("nan"), 0])):
            pick = dict(self.pick, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_target(self.image([self.xy]), self.cam.j, pick, self.bins)


if __name__ == '__main__':
    unittest.main()
