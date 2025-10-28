import numpy as np

def calculate_angle(a: np.ndarray | None, b: np.ndarray | None, c: np.ndarray | None) -> float:
    """
    Calculates the angle (in radians) between three 2D or 3D points at vertex b.
    
    The angle is formed by the vectors ba (from b to a) and bc (from b to c).
    
    Args:
        a: A numpy array representing point 'a'.
        b: A numpy array representing point 'b' (the vertex).
        c: A numpy array representing point 'c'.
        
    Returns:
        The angle in radians, or 0.0 if the angle cannot be computed
        (e.g., if vectors have zero length).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Handle zero-length vectors to avoid division by zero
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    dot_product = np.dot(ba, bc)
    
    cosine_angle = dot_product / (norm_ba * norm_bc)
    
    # Clip to handle potential floating-point inaccuracies
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cosine_angle)

    return angle_rad