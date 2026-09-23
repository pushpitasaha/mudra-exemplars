# Hierarchical pose errors: joint -> finger -> whole hand

import numpy as np


# MediaPipe landmark names in landmark-index order
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


# Landmarks belonging to each finger
FINGER_LANDMARKS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}


def as_hand(pose):
    """Convert 63D or 21x3 pose to 21x3."""
    pose = np.asarray(pose, dtype=np.float32)

    if pose.shape == (63,):
        return pose.reshape(21, 3)

    if pose.shape == (21, 3):
        return pose

    raise ValueError(
        f"Expected shape (63,) or (21, 3), got {pose.shape}"
    )


def joint_errors(live_pose, target_pose):
    """Euclidean error for each of the 21 landmarks."""
    live = as_hand(live_pose)
    target = as_hand(target_pose)

    return np.linalg.norm(live - target, axis=1)


def finger_errors(joint_err):
    """Mean landmark error for each finger."""
    joint_err = np.asarray(joint_err, dtype=np.float32)

    if joint_err.shape != (21,):
        raise ValueError(
            f"Expected 21 joint errors, got {joint_err.shape}"
        )

    errors = {}

    for finger, indices in FINGER_LANDMARKS.items():
        errors[finger] = float(np.mean(joint_err[indices]))

    return errors


def hand_error(joint_err):
    """Mean error across the 21 hand landmarks."""
    joint_err = np.asarray(joint_err, dtype=np.float32)

    if joint_err.shape != (21,):
        raise ValueError(
            f"Expected 21 joint errors, got {joint_err.shape}"
        )

    return float(np.mean(joint_err))


def correction_vectors(live_pose, target_pose):
    """
    Direction from each live landmark toward its target.
    vector = target - live
    """
    live = as_hand(live_pose)
    target = as_hand(target_pose)

    return target - live


def analyze_pose_error(live_pose, target_pose):
    """Return all hierarchical error information."""
    joints = joint_errors(live_pose, target_pose)
    fingers = finger_errors(joints)
    whole_hand = hand_error(joints)
    vectors = correction_vectors(live_pose, target_pose)

    worst_joint_idx = int(np.argmax(joints))
    worst_finger = max(fingers, key=fingers.get)

    return {
        "joint_errors": joints,
        "finger_errors": fingers,
        "hand_error": whole_hand,
        "correction_vectors": vectors,

        "worst_joint_index": worst_joint_idx,
        "worst_joint_name": LANDMARK_NAMES[worst_joint_idx],
        "worst_joint_error": float(joints[worst_joint_idx]),

        "worst_finger": worst_finger,
        "worst_finger_error": fingers[worst_finger],
    }