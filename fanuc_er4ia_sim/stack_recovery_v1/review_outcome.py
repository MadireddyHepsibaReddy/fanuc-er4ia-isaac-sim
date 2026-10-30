"""Explicit outcome for a recorded stop before any physical grasp attempt."""
import json
from pathlib import Path


class PreGraspReview(RuntimeError):
    pass


def record_pregrasp_review(run):
    run=Path(run)
    request_file=run/'human_review_request.json'
    status_file=run/'recovery_status.json'
    if not request_file.is_file() or not status_file.is_file():
        return None
    request=json.loads(request_file.read_text())
    status=json.loads(status_file.read_text())
    if request.get('grasp_attempted') is not False or status.get('decision')!='human_review_requested':
        return None
    report=dict(report_kind='pre_grasp_review_stop', outcome='human_review_requested',
                task_complete=False, execution_started=False, physical_execution_evaluated=False,
                verified_pickups=0, verified_placements=0, recovery_success=False,
                recovery_attempted=bool(status.get('recovery_attempted')),
                recovery_applied=bool(status.get('recovery_applied')),
                abort_reason=request['reason'], recovery_status=status,
                same_simulator_session=True, human_requests_are_local_records=True)
    path=run/'execution_report.json'
    # Existing physical execution reports must never be replaced by this summary.
    with path.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    return report
