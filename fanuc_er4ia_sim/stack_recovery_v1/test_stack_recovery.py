import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import cv2,numpy as np
import scene_config as C
import test_cube_pose
from plan_sort import build_plan
from live_recovery import validate_update,before_first_pick,HumanReviewRequired,validate_explanation
from stack_shift import validate_request,adjusted_labels
from patch_stack_executor import patch_executor


class StackRecoveryTests(unittest.TestCase):
    def setUp(self):
        fixture=test_cube_pose.CubePoseTests();fixture.setUp()
        self.original=build_plan({'objects':fixture.inventory()},fixture.cam,task='cube_stack')
        for p in self.original['picks']:p['pose_silhouette_iou']=.96
        self.candidate=copy.deepcopy(self.original)
        self.candidate['picks'][0]['xy'][0]+=.010
        target=self.candidate['picks'][0]
        self.evidence=dict(status='target_moved',candidate_count=1,measured_xy_m=target['xy'].copy(),image_fit_yaw_deg=target['yaw_deg'])
        self.state={'objects':{'SortCube0':{'position':[*self.original['picks'][0]['xy'],.8949],'orientation':[1,0,0,0]},
                              'SortCube1':{'position':[.42,.18,.8949],'orientation':[1,0,0,0]}}}

    def test_accepts_image_supported_update_and_preserves_original(self):
        before=copy.deepcopy(self.original)
        self.assertAlmostEqual(validate_update(self.original,self.candidate,self.evidence)['change_mm'],10)
        self.assertEqual(self.original,before)

    def test_preserves_confidence_and_fit_gates(self):
        for name,value in [('pose_confidence',.2),('pose_silhouette_iou',.8)]:
            p=copy.deepcopy(self.candidate);p['picks'][0][name]=value
            with self.assertRaises(ValueError):validate_update(self.original,p,self.evidence)

    def test_rejects_changed_stack_colour_order(self):
        self.candidate['picks'][0]['colour']='red'
        with self.assertRaises(ValueError):validate_update(self.original,self.candidate,self.evidence)

    def test_rejects_changed_destination(self):
        self.candidate['picks'][0]['place_xy']=[.1,.1]
        with self.assertRaises(ValueError):validate_update(self.original,self.candidate,self.evidence)

    def test_rejects_other_cube_motion(self):
        self.candidate['picks'][1]['xy'][0]+=.005
        with self.assertRaises(ValueError):validate_update(self.original,self.candidate,self.evidence)

    def test_rejects_uncertain_and_ambiguous_observation(self):
        for fields in ({'status':'uncertain'},{'candidate_count':2}):
            with self.assertRaises(ValueError):validate_update(self.original,self.candidate,dict(self.evidence,**fields))

    def test_rejects_yaw_disagreement(self):
        with self.assertRaises(ValueError):validate_update(self.original,self.candidate,dict(self.evidence,image_fit_yaw_deg=20))

    def test_rejects_position_disagreement_or_nan(self):
        for xy in ([.1,.1],[float('nan'),0]):
            with self.assertRaises(ValueError):validate_update(self.original,self.candidate,dict(self.evidence,measured_xy_m=xy))

    def test_rejects_wrong_task_and_incomplete_inventory(self):
        for change in ('task','inventory'):
            p=copy.deepcopy(self.candidate)
            if change=='task':p['task']='sort'
            else:p['picks'].pop()
            with self.assertRaises(ValueError):validate_update(self.original,p,self.evidence)

    def test_control_and_shift_setup_preserve_scene_inputs(self):
        before=copy.deepcopy(self.state)
        for delta in ([0,0,0],[.01,0,0]):
            r=validate_request(self.state,self.original,'SortCube0',delta)
            np.testing.assert_allclose(r['requested_position_m'],np.array(before['objects']['SortCube0']['position'])+delta)
        self.assertEqual(before,self.state)

    def test_rejects_unrequested_translations(self):
        for delta in ([0,.01,0],[.02,0,0],[.01,0,.01],[float('nan'),0,0]):
            with self.assertRaises(ValueError):validate_request(self.state,self.original,'SortCube0',delta)

    def test_rejects_nearby_cube_destination(self):
        self.state['objects']['SortCube1']['position']=list(np.array(self.state['objects']['SortCube0']['position'])+[.012,0,0])
        with self.assertRaises(ValueError):validate_request(self.state,self.original,'SortCube0',[.01,0,0])

    def test_rejects_wrong_authored_target(self):
        with self.assertRaises(ValueError):validate_request(self.state,self.original,'SortCube1',[.01,0,0])

    def test_truth_label_adjustment_is_separate_and_non_mutating(self):
        truth={'objects':[{'name':'SortCube0','xyz':[.36,.07,.005]}]};before=copy.deepcopy(truth)
        r=adjusted_labels(truth,'SortCube0',[.01,0,0]);self.assertAlmostEqual(r['objects'][0]['xyz'][0],.37)
        self.assertEqual(truth,before)

    def test_runtime_patch_keeps_stack_support_and_refuses_different_source(self):
        import sort_scene
        raw=Path(sort_scene.__file__).read_bytes()
        result=patch_executor(raw,hashlib.sha256(raw).hexdigest())
        self.assertIn('verify_cube_stack(',result)
        self.assertIn('not event.get("stack_support_verified")',result)
        self.assertIn('before_first_pick(run, plan, cam',result)
        with self.assertRaises(ValueError):patch_executor(raw+b'\n',hashlib.sha256(raw).hexdigest())

    def decision(self,folder,mode,status):
        root=Path(folder);(root/'live_settings.json').write_text(json.dumps({'mode':mode}))
        path=root/'test.png';cv2.imwrite(str(path),np.full((8,8,3),150,np.uint8))
        with patch('stack_observation.save_observation',return_value={'image':str(path)}),patch('live_recovery.check_target',return_value={'status':status}),patch('live_recovery.child') as worker:
            result=before_first_pick(root,self.original,Mock(),{},Mock(),Mock(),None)
            worker.assert_not_called();return result

    def test_disabled_mode_never_calls_new_model(self):
        with tempfile.TemporaryDirectory() as f:self.assertEqual(self.decision(f,'disabled','target_moved'),self.original)

    def test_unchanged_control_retains_plan_without_model(self):
        with tempfile.TemporaryDirectory() as f:self.assertEqual(self.decision(f,'enabled','position_consistent'),self.original)

    def test_uncertain_observation_records_review_before_grasp(self):
        with tempfile.TemporaryDirectory() as f:
            with self.assertRaises(HumanReviewRequired):self.decision(f,'enabled','uncertain')
            r=json.loads((Path(f)/'human_review_request.json').read_text());self.assertFalse(r['grasp_attempted']);self.assertFalse(r['automatically_sent'])

    def test_ai_explanation_must_not_invent_a_cause(self):
        r=dict(observed_change='target_position_changed',cause='robot_collision',grasp_attempted=False,recommended_action='replan',explanation='Unsupported cause')
        with self.assertRaises(ValueError):validate_explanation(r)
