import json
from pathlib import Path
import tempfile
import unittest

from guarded_refinement import prefer_measured_pose
from review_outcome import record_pregrasp_review


class RevisionV3Tests(unittest.TestCase):
    def test_equal_fit_keeps_supported_measurement(self):
        self.assertTrue(prefer_measured_pose(([.36,.07],-2,.96),([.3601,.07],1.5,.96),[.36,.07],-2))

    def test_better_measurement_is_kept_but_better_refinement_is_not_discarded(self):
        measured=([.36,.07],-2,.96)
        self.assertTrue(prefer_measured_pose(measured,([.36,.07],1.5,.95),[.36,.07],-2))
        self.assertFalse(prefer_measured_pose(measured,([.36,.07],1.5,.97),[.36,.07],-2))

    def test_retention_keeps_quality_bounds_and_finite_checks(self):
        for value in [([.36,.07],-2,.84),([.365,.07],-2,.97),
                      ([.36,.07],20,.97),([float('nan'),.07],-2,.97)]:
            with self.subTest(value=value):
                self.assertFalse(prefer_measured_pose(value,([.36,.07],-2,.8),[.36,.07],-2))

    def write_request(self,root,grasp=False,decision='human_review_requested'):
        (root/'human_review_request.json').write_text(json.dumps({'reason':'yaw disagreement','grasp_attempted':grasp}))
        (root/'recovery_status.json').write_text(json.dumps({'decision':decision,'recovery_attempted':True,'recovery_applied':False}))

    def test_review_report_records_zero_attempts_without_claiming_physics_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.write_request(root)
            result=record_pregrasp_review(root)
            self.assertFalse(result['task_complete'])
            self.assertFalse(result['physical_execution_evaluated'])
            self.assertFalse(result['execution_started'])
            self.assertEqual(result['verified_pickups'],0)
            self.assertEqual(result['outcome'],'human_review_requested')
            self.assertEqual(result['abort_reason'],'yaw disagreement')

    def test_missing_records_are_not_fabricated_as_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.assertIsNone(record_pregrasp_review(root))
            self.assertFalse((root/'execution_report.json').exists())

    def test_post_grasp_or_unclassified_error_is_not_pregrasp_review(self):
        for grasp,decision in [(True,'human_review_requested'),(False,'other_error')]:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);self.write_request(root,grasp,decision)
                self.assertIsNone(record_pregrasp_review(root))
                self.assertFalse((root/'execution_report.json').exists())

    def test_existing_execution_evidence_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.write_request(root)
            path=root/'execution_report.json';path.write_text('original report')
            with self.assertRaises(FileExistsError):
                record_pregrasp_review(root)
            self.assertEqual(path.read_text(),'original report')
