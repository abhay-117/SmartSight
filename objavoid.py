def determine_avoidance_path(detected_objects, frame_width):
    left_blocked = False
    right_blocked = False
    center_blocked = False

    for obj, position in detected_objects.items():
        if position == "on the left":
            left_blocked = True
        elif position == "on the right":
            right_blocked = True
        elif position == "in the center":
            center_blocked = True

    # If center is blocked but left and right are open, announce both sides as usable
    if center_blocked and not left_blocked and not right_blocked:
        return "both left and right are usable"

    # If center is blocked, prioritize moving sideways
    if center_blocked:
        if not left_blocked:
            return "left"
        elif not right_blocked:
            return "right"
        else:
            return "back"

    # If only left is blocked, move right
    if left_blocked and not right_blocked:
        return "right"

    # If only right is blocked, move left
    if right_blocked and not left_blocked:
        return "left"

    return "safe"
