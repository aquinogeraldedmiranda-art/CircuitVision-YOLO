# CircuitVision website and private progress records

## Public user experience

CircuitVision presents a single **YOLOv11** detection system. The public
interface focuses on Home, Image Detection, Live Camera, Results, Component Guide
and a concise About page. Do **not** display model rounds, teammate names,
notebook stages, research plans, commits, experimental availability, or internal
progress updates to website users.

Avoid claims that electrical values are measured, that every part will be
identified, or that the model has achieved test accuracy not supported by actual
evaluation. The public interface can truthfully describe visual predictions,
bounding boxes, labels and confidence scores.

## Private engineering records

The `project_status.json` file in this GitHub repository preserves internal
model provenance and development milestones for the team, including details
required for the IEEE paper. It is **not served through the public website**.
The public `/api/project-status` endpoint was removed intentionally.

Every substantial change to `app.js`, backend code, models, notebooks or
camera tracking must include a review of `project_status.json` in the same
pull request. GitHub Actions enforces this rule.

For each update, document what was prepared, tested, deployed or validated,
without inflating results. Keep checkpoint versions for safe rollback, use
independent holdout evaluation for new trained models, and confirm the GitHub
revision is live on the **existing** Render service.

The user-facing brand is always **CircuitVision · YOLOv11**, even if the team
keeps several experimental detectors internally while preparing a validated
jointly trained checkpoint.

## Website
https://circuitvision-yolo-lab.onrender.com/
