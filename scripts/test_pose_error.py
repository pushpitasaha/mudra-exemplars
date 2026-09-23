import numpy as np

from pose_error import analyze_pose_error


# Fake target hand
target = np.zeros((21, 3), dtype=np.float32)

# Live hand starts identical
live = target.copy()

# Move only index fingertip
live[8] = [0.3, 0.0, 0.0]


result = analyze_pose_error(live, target)


print("Hand error:")
print(result["hand_error"])

print("\nFinger errors:")
for finger, error in result["finger_errors"].items():
    print(f"{finger:7s}: {error:.4f}")

print("\nWorst finger:")
print(result["worst_finger"])

print("\nWorst joint:")
print(result["worst_joint_name"])

print("\nWorst joint error:")
print(result["worst_joint_error"])

print("\nCorrection vector:")
print(result["correction_vectors"][8])