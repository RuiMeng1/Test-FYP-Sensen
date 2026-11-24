"""Pose and visible-phone evidence for the offline office activity pass."""
import math
from pathlib import Path


def box_iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(1, area_a + area_b - intersection)


def angle(a, b, c):
    """Joint angle in degrees, or None for degenerate landmarks."""
    u = (a[0] - b[0], a[1] - b[1])
    v = (c[0] - b[0], c[1] - b[1])
    size = math.hypot(*u) * math.hypot(*v)
    if size < 1:
        return None
    return math.degrees(math.acos(max(-1, min(1, (u[0] * v[0] + u[1] * v[1]) / size))))


def posture_from_keypoints(points):
    """Return a conservative 2D posture suggestion and keypoint confidence."""
    candidates = []
    for shoulder, hip, knee, ankle in ((5, 11, 13, 15), (6, 12, 14, 16)):
        joints = [points[index] for index in (shoulder, hip, knee, ankle)]
        if any(len(point) < 3 or point[2] < .35 for point in joints):
            continue
        hip_angle = angle(joints[0], joints[1], joints[2])
        knee_angle = angle(joints[1], joints[2], joints[3])
        if hip_angle is None or knee_angle is None:
            continue
        score = min(float(point[2]) for point in joints)
        if 55 <= hip_angle <= 145 and 60 <= knee_angle <= 145:
            candidates.append(('sitting', score))
        elif hip_angle >= 150 and knee_angle >= 155:
            candidates.append(('standing', score))
    labels = {label for label, _ in candidates}
    if len(labels) != 1:
        return None, None
    label = labels.pop()
    return label, max(score for found, score in candidates if found == label)


def phone_near_hand(phone_box, person_box, points):
    """A visible COCO phone close to a visible wrist inside the person's box."""
    center = ((phone_box[0] + phone_box[2]) / 2, (phone_box[1] + phone_box[3]) / 2)
    if not (person_box[0] <= center[0] <= person_box[2] and person_box[1] <= center[1] <= person_box[3]):
        return False
    height = max(1, person_box[3] - person_box[1])
    return any(len(points[index]) >= 3 and points[index][2] >= .35
               and math.dist(center, points[index][:2]) <= .22 * height
               for index in (9, 10))


def chair_support(person_box, points, chairs):
    """Require a detected seat overlapping the bent person's visible hip."""
    hips = [points[index] for index in (11, 12)
            if len(points[index]) >= 3 and points[index][2] >= .35]
    if not hips:
        return False
    height = max(1, person_box[3] - person_box[1])
    for chair in chairs:
        if box_iou(person_box, chair) < .08:
            continue
        margin = .1 * (chair[2] - chair[0])
        if any(chair[0] - margin <= hip[0] <= chair[2] + margin
               and chair[1] - .1 * height <= hip[1] <= chair[3] for hip in hips):
            return True
    return False


class OfficeVision:
    """Run two small YOLO models on each sampled frame; no extra video storage."""

    def __init__(self, device='cpu', pose_model='yolov8n-pose.pt',
                 object_model='yolov8n.pt', object_size=1920):
        from ultralytics import YOLO
        self.device = device
        self.object_size = object_size
        bundled = Path(__file__).resolve().parents[1] / 'models' / 'yolov8n-pose.pt'
        if pose_model == 'yolov8n-pose.pt' and bundled.exists():
            pose_model = str(bundled)
        try:
            self.pose = YOLO(pose_model)
            self.objects = YOLO(object_model)
        except Exception as error:
            raise RuntimeError('Could not load office vision checkpoints. Check model paths or download access.') from error
        if self.pose.task != 'pose' or self.objects.task != 'detect':
            raise ValueError('Expected a pose checkpoint and a COCO object detection checkpoint')
        names = self.objects.names
        if names.get(56) != 'chair' or names.get(57) != 'couch' or names.get(67) != 'cell phone':
            raise ValueError('Object checkpoint must use COCO chair, couch, and cell phone classes')

    def predict(self, frame, boxes):
        pose = self.pose.predict(frame, conf=.25, imgsz=960, device=self.device, verbose=False)[0]
        objects = self.objects.predict(frame, classes=[56, 57, 67], conf=.20, imgsz=self.object_size,
                                       device=self.device, verbose=False)[0]
        pose_boxes = pose.boxes.xyxy.cpu().numpy() if pose.boxes is not None else []
        points = pose.keypoints.data.cpu().numpy() if pose.keypoints is not None else []
        object_boxes = objects.boxes.xyxy.cpu().numpy() if objects.boxes is not None else []
        object_scores = objects.boxes.conf.cpu().numpy() if objects.boxes is not None else []
        object_classes = objects.boxes.cls.cpu().numpy() if objects.boxes is not None else []
        phones = [(box, score) for box, score, cls in zip(object_boxes, object_scores, object_classes)
                  if int(cls) == 67]
        chairs = [box for box, cls in zip(object_boxes, object_classes) if int(cls) in (56, 57)]
        candidates = sorted(((box_iou(box, pose_box), i, j)
                             for i, box in enumerate(boxes)
                             for j, pose_box in enumerate(pose_boxes)), reverse=True)
        matches, used_pose = {}, set()
        for overlap, i, j in candidates:
            if overlap < .35:
                break
            if i not in matches and j not in used_pose:
                matches[i] = j
                used_pose.add(j)
        observations = []
        for i, box in enumerate(boxes):
            observation = {'posture': None, 'posture_score': None, 'phone_score': None}
            if i in matches:
                landmarks = points[matches[i]]
                observation['posture'], observation['posture_score'] = posture_from_keypoints(landmarks)
                if observation['posture'] == 'sitting' and not chair_support(box, landmarks, chairs):
                    observation['posture'], observation['posture_score'] = None, None
                nearby = [float(score) for phone, score in phones
                          if phone_near_hand(phone, box, landmarks)]
                if nearby:
                    observation['phone_score'] = max(nearby)
            observations.append(observation)
        return observations
