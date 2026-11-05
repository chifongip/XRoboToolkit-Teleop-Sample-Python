"""Omnihand Pro trigger targets for control and articulated collision checks."""

# Left-hand targets; right thumb roll and abduction change sign.
HAND_CLOSED = {
    "thumb_roll": -0.6,
    "thumb_abad": 0.6,
    # Deep MCP flexion pushes the thumb into the curled fingers. Keep the
    # proximal segment outside them and curl primarily at the distal joints.
    "thumb_mcp": -0.05,
    "thumb_pip": -0.6,
    "index_abad": 0.0,
    "index_mcp": 1.0,
    "index_pip": 1.0,
    "middle_abad": 0.0,
    "middle_mcp": 1.0,
    "middle_pip": 1.0,
    "ring_mcp": 1.0,
    "pinky_mcp": 1.0,
}
