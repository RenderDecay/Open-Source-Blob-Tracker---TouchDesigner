import numpy as np
import cv2
from collections import deque

# Globals
if 'avg_y_history' not in globals():
    avg_y_history = []

# Persistent cache
if '_detector_cache' not in globals():
    _detector_cache = {'detector': None, 'last_params': None}

if '_frame_skip_cache' not in globals():
    _frame_skip_cache = {
        'counter': 0, 
        'last_keypoints': [], 
        'last_centers': [],
        'last_sizes': [],
        'velocities': [],
        'smoothed_centers': [],
        'smoothed_velocities': [],
        'smoothed_sizes': [],
        'blob_ids': [],
        'blob_confidence': {}  # Track how stable each blob is
    }

if '_image_cache' not in globals():
    _image_cache = {'out_img': None, 'blobs_img': None, 'size': None}

if '_trail_cache' not in globals():
    _trail_cache = {'interpolated': None, 'source_centers': None}

# Pre-computed constants
_BASIS_MATRIX = 0.5 * np.array([
    [0, 2, 0, 0],
    [-1, 0, 1, 0],
    [2, -5, 4, -1],
    [-1, 3, -3, 1]
], dtype=np.float32)

def get_basis_matrix(resolution):
    """Generate basis matrix for given resolution"""
    t = np.linspace(0, 1, resolution, dtype=np.float32)
    t2 = t * t
    t3 = t2 * t
    return np.column_stack([
        np.ones(resolution, dtype=np.float32), 
        t, 
        t2, 
        t3
    ]) @ _BASIS_MATRIX

def interpolate_catmull_rom_cached(centers, resolution=8):
    global _trail_cache
    
    if len(centers) < 4:
        return centers
    
    cache_key = (tuple(map(tuple, centers)), resolution)
    if _trail_cache.get('cache_key') == cache_key:
        return _trail_cache['interpolated']
    
    points_array = np.array(centers, dtype=np.float32)
    basis_matrix = get_basis_matrix(resolution)
    result = []
    
    for i in range(1, len(centers) - 2):
        control_points = points_array[i-1:i+3]
        segment = basis_matrix @ control_points
        result.extend(segment.tolist())
    
    _trail_cache['interpolated'] = result
    _trail_cache['cache_key'] = cache_key
    
    return result

def match_blobs_fast(current_centers, current_sizes, previous_centers, previous_ids, max_distance=120):
    """
    Fast vectorized blob matching using numpy
    """
    if not previous_centers or not current_centers:
        new_ids = list(range(len(current_centers)))
        return current_centers, current_sizes, new_ids
    
    # Convert to numpy for vectorized operations
    curr_arr = np.array(current_centers, dtype=np.float32)
    prev_arr = np.array(previous_centers, dtype=np.float32)
    
    # Calculate SQUARED distance matrix (skip sqrt for comparison)
    diff = curr_arr[:, None, :] - prev_arr[None, :, :]
    distances_sq = np.sum(diff * diff, axis=2)
    max_dist_sq = max_distance * max_distance
    
    matched_centers = []
    matched_sizes = []
    matched_ids = []
    used_previous = set()
    next_new_id = max(previous_ids) + 1 if previous_ids else 0
    
    # For each current blob, find closest previous blob
    for i in range(len(current_centers)):
        dists_sq_to_prev = distances_sq[i]
        
        best_dist_sq = max_dist_sq
        best_idx = -1
        
        for prev_idx in range(len(previous_centers)):
            if prev_idx in used_previous:
                continue
            if dists_sq_to_prev[prev_idx] < best_dist_sq:
                best_dist_sq = dists_sq_to_prev[prev_idx]
                best_idx = prev_idx
        
        if best_idx >= 0:
            matched_centers.append(current_centers[i])
            matched_sizes.append(current_sizes[i])
            matched_ids.append(previous_ids[best_idx])
            used_previous.add(best_idx)
        else:
            matched_centers.append(current_centers[i])
            matched_sizes.append(current_sizes[i])
            matched_ids.append(next_new_id)
            next_new_id += 1
    
    return matched_centers, matched_sizes, matched_ids

def draw_dotted_line(img, pt1, pt2, color, thickness, gap=8):
    """Draw a dotted line between two points"""
    dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    
    if dist == 0:
        return
    
    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist
    
    current_dist = 0
    dash_length = gap // 2
    
    while current_dist < dist:
        x1 = int(pt1[0] + dx * current_dist)
        y1 = int(pt1[1] + dy * current_dist)
        
        end_dist = min(current_dist + dash_length, dist)
        x2 = int(pt1[0] + dx * end_dist)
        y2 = int(pt1[1] + dy * end_dist)
        
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_4)
        
        current_dist += gap

def onCook(scriptOp):
    global _detector_cache, _frame_skip_cache, _image_cache
    
    if scriptOp.inputs[0] is None or scriptOp.inputs[1] is None:
        return

    input_thresh = scriptOp.inputs[0]
    input_res_ref = scriptOp.inputs[1]

    img_thresh = input_thresh.numpyArray(delayed=False)
    img_ref = input_res_ref.numpyArray(delayed=False)

    if img_thresh is None or img_thresh.size == 0 or img_ref is None or img_ref.size == 0:
        return

    img_thresh = np.flipud(img_thresh)
    img_ref = np.flipud(img_ref)

    out_h, out_w = img_ref.shape[:2]

    if img_thresh.ndim == 3 and img_thresh.shape[2] >= 1:
        thresh_gray = np.ascontiguousarray((img_thresh[:, :, 0] * 255).astype(np.uint8))
    else:
        return

    comp = parent()

    # Read parameters
    outline_col = (255, 255, 255)
    try:
        outline_col = (
            int(comp.par.Outlinecolr.eval() * 255),
            int(comp.par.Outlinecolg.eval() * 255),
            int(comp.par.Outlinecolb.eval() * 255),
        )
    except:
        pass
    
    trail_col = (255, 255, 255)
    try:
        trail_col = (
            int(comp.par.Trailcolr.eval() * 255),
            int(comp.par.Trailcolg.eval() * 255),
            int(comp.par.Trailcolb.eval() * 255),
        )
    except:
        pass

    min_area = float(comp.par.Minarea.eval())
    max_area = max(min_area + 1, float(comp.par.Maxarea.eval()))
    max_blobs = int(comp.par.Maxblobs.eval()) if hasattr(comp.par, 'Maxblobs') else 100
    show_ids = comp.par.Showids.eval()
    show_xy = comp.par.Showxy.eval() if hasattr(comp.par, 'Showxy') else False
    draw_trails = comp.par.Drawtrails.eval()
    line_smoothness = int(comp.par.Linesmoothness.eval()) if hasattr(comp.par, 'Linesmoothness') else 8
    line_smoothness = max(2, min(16, line_smoothness))
    use_dotted = comp.par.Usedotted.eval() if hasattr(comp.par, 'Usedotted') else False
    max_line_length_norm = float(comp.par.Maxlinelength.eval()) if hasattr(comp.par, 'Maxlinelength') else 1.0
    blob_thickness = int(comp.par.Blobthickness.eval()) if hasattr(comp.par, 'Blobthickness') else -1
    blob_thickness = max(1, blob_thickness)
    
    draw_connections = comp.par.Drawconnections.eval() if hasattr(comp.par, 'Drawconnections') else False
    show_grid = comp.par.Showgrid.eval() if hasattr(comp.par, 'Showgrid') else False
    grid_spacing = float(comp.par.Gridspacing.eval()) if hasattr(comp.par, 'Gridspacing') else 50.0
    show_leaders = comp.par.Showleaders.eval() if hasattr(comp.par, 'Showleaders') else False
    show_metrics = comp.par.Showmetrics.eval() if hasattr(comp.par, 'Showmetrics') else False
    use_brackets = comp.par.Usebrackets.eval() if hasattr(comp.par, 'Usebrackets') else False
    bracket_length = float(comp.par.Bracketlength.eval()) if hasattr(comp.par, 'Bracketlength') else 0.3
    bracket_length = np.clip(bracket_length, 0.1, 0.5)
    
    enable_skip = comp.par.Enableframeskip.eval() if hasattr(comp.par, 'Enableframeskip') else False
    frame_skip_interval = int(comp.par.Frameskipinterval.eval()) if hasattr(comp.par, 'Frameskipinterval') else 2
    frame_skip_interval = max(2, min(10, frame_skip_interval))
    
    motion_smoothing = float(comp.par.Motionsmoothing.eval()) if hasattr(comp.par, 'Motionsmoothing') else 0.5
    motion_smoothing = np.clip(motion_smoothing, 0.0, 1.0)
    
    size_smoothing = float(comp.par.Sizesmoothing.eval()) if hasattr(comp.par, 'Sizesmoothing') else 0.5
    size_smoothing = np.clip(size_smoothing, 0.0, 1.0)
    
    resolution_scale = float(comp.par.Resolutionscale.eval()) if hasattr(comp.par, 'Resolutionscale') else 1.0
    resolution_scale = np.clip(resolution_scale, 0.1, 1.0)

    # Resolution scaling
    h_src, w_src = thresh_gray.shape[:2]
    
    if resolution_scale < 1.0:
        detect_w = max(1, int(w_src * resolution_scale))
        detect_h = max(1, int(h_src * resolution_scale))
        thresh_detect = cv2.resize(thresh_gray, (detect_w, detect_h), interpolation=cv2.INTER_AREA)
    else:
        thresh_detect = thresh_gray
        detect_w, detect_h = w_src, h_src
    
    # Scale from detection resolution to OUTPUT resolution
    scale_x = out_w / detect_w
    scale_y = out_h / detect_h
    scale_avg = (scale_x + scale_y) * 0.5

    # Frame skip logic
    _frame_skip_cache['counter'] += 1
    should_detect = (not enable_skip) or (_frame_skip_cache['counter'] % frame_skip_interval == 0)
    
    # Detect blobs
    if should_detect:
        # Adjust area thresholds for detection resolution
        area_scale = resolution_scale * resolution_scale
        adjusted_min = min_area * area_scale
        adjusted_max = max_area * area_scale
        
        params_tuple = (adjusted_min, adjusted_max)
        if _detector_cache['detector'] is None or _detector_cache['last_params'] != params_tuple:
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = adjusted_min
            params.maxArea = adjusted_max
            params.filterByCircularity = False
            params.filterByConvexity = False
            params.filterByInertia = False
            params.minThreshold = 1
            params.maxThreshold = 255
            _detector_cache['detector'] = cv2.SimpleBlobDetector_create(params)
            _detector_cache['last_params'] = params_tuple

        keypoints = _detector_cache['detector'].detect(thresh_detect)
        
        if len(keypoints) > max_blobs:
            keypoints = sorted(keypoints, key=lambda kp: kp.size, reverse=True)[:max_blobs]
        
        # Scale keypoints to output resolution
        raw_centers = []
        raw_sizes = []
        for kp in keypoints:
            cx = int(kp.pt[0] * scale_x)
            cy = int(kp.pt[1] * scale_y)
            size = int(kp.size * scale_avg)
            raw_centers.append((cx, cy))
            raw_sizes.append(size)
        
        # Match blobs - moderate distance for stability
        matched_centers, matched_sizes, matched_ids = match_blobs_fast(
            raw_centers, 
            raw_sizes,
            _frame_skip_cache.get('smoothed_centers', []),
            _frame_skip_cache.get('blob_ids', []),
            max_distance=120  # Balanced - not too tight, not too loose
        )
        
        # Update confidence scores efficiently
        confidence_dict = _frame_skip_cache.get('blob_confidence', {})
        
        for blob_id in matched_ids:
            if blob_id in confidence_dict:
                confidence_dict[blob_id] = min(10, confidence_dict[blob_id] + 1)
            else:
                confidence_dict[blob_id] = 1
        
        # Clean up old blob IDs not in current frame
        if len(confidence_dict) > max_blobs * 2:
            confidence_dict = {bid: conf for bid, conf in confidence_dict.items() if bid in matched_ids}
        
        # Apply smoothing - responsive but stable
        if _frame_skip_cache.get('smoothed_centers') and len(_frame_skip_cache['smoothed_centers']) > 0:
            prev_smooth_by_id = {}
            prev_size_by_id = {}
            for i, blob_id in enumerate(_frame_skip_cache.get('blob_ids', [])):
                if i < len(_frame_skip_cache['smoothed_centers']):
                    prev_smooth_by_id[blob_id] = _frame_skip_cache['smoothed_centers'][i]
                if i < len(_frame_skip_cache['smoothed_sizes']):
                    prev_size_by_id[blob_id] = _frame_skip_cache['smoothed_sizes'][i]
            
            smoothed_centers = []
            smoothed_sizes = []
            
            # Cache motion smoothing multipliers
            alpha_low = 0.2 + (motion_smoothing * 0.2)  # 0.2-0.4
            alpha_high = 0.4 + (motion_smoothing * 0.3)  # 0.4-0.7
            size_alpha_low = 0.2 + (size_smoothing * 0.2)
            size_alpha_high = 0.4 + (size_smoothing * 0.3)
            
            for center, size, blob_id in zip(matched_centers, matched_sizes, matched_ids):
                # Adaptive smoothing based on confidence
                blob_conf = confidence_dict.get(blob_id, 1)
                
                # Use pre-calculated alphas
                if blob_conf >= 3:
                    smooth_alpha = alpha_high
                    size_alpha = size_alpha_high
                else:
                    smooth_alpha = alpha_low
                    size_alpha = size_alpha_low
                
                if blob_id in prev_smooth_by_id:
                    prev_center = prev_smooth_by_id[blob_id]
                    smooth_x = int(smooth_alpha * center[0] + (1 - smooth_alpha) * prev_center[0])
                    smooth_y = int(smooth_alpha * center[1] + (1 - smooth_alpha) * prev_center[1])
                    smoothed_centers.append((smooth_x, smooth_y))
                else:
                    smoothed_centers.append(center)
                
                if blob_id in prev_size_by_id:
                    prev_size = prev_size_by_id[blob_id]
                    smooth_size = int(size_alpha * size + (1 - size_alpha) * prev_size)
                    smoothed_sizes.append(smooth_size)
                else:
                    smoothed_sizes.append(size)
            
            centers = smoothed_centers
            sizes = smoothed_sizes
        else:
            centers = matched_centers
            sizes = matched_sizes
        
        # Store for next frame
        _frame_skip_cache['smoothed_centers'] = centers
        _frame_skip_cache['smoothed_sizes'] = sizes
        _frame_skip_cache['blob_ids'] = matched_ids
        _frame_skip_cache['last_centers'] = raw_centers
        _frame_skip_cache['last_sizes'] = raw_sizes
        
    else:
        # Use smoothed positions from last detection
        centers = _frame_skip_cache.get('smoothed_centers', [])
        sizes = _frame_skip_cache.get('smoothed_sizes', [])

    # Initialize output images
    if (_image_cache['out_img'] is None or _image_cache['size'] != (out_h, out_w)):
        _image_cache['out_img'] = np.zeros((out_h, out_w, 4), dtype=np.float32)
        _image_cache['blobs_img'] = np.zeros((out_h, out_w, 4), dtype=np.float32)
        _image_cache['size'] = (out_h, out_w)
    else:
        _image_cache['out_img'].fill(0)
        _image_cache['blobs_img'].fill(0)
    
    out_img = _image_cache['out_img']
    blobs_img = _image_cache['blobs_img']

    outline_col_norm = (outline_col[0]/255.0, outline_col[1]/255.0, outline_col[2]/255.0, 1.0)
    trail_col_norm = (trail_col[0]/255.0, trail_col[1]/255.0, trail_col[2]/255.0, 1.0)
    white = (1.0, 1.0, 1.0, 1.0)

    num_blobs = len(centers)
    
    # Draw blobs
    if num_blobs > 0:
        centers_x = np.array([c[0] for c in centers], dtype=np.int32)
        centers_y = np.array([c[1] for c in centers], dtype=np.int32)
        sizes_arr = np.array(sizes, dtype=np.int32)
        half_sizes = (sizes_arr * 0.5).astype(np.int32)
        
        x0_arr = np.clip(centers_x - half_sizes, 0, out_w)
        y0_arr = np.clip(centers_y - half_sizes, 0, out_h)
        x1_arr = np.clip(centers_x + half_sizes, 0, out_w)
        y1_arr = np.clip(centers_y + half_sizes, 0, out_h)
        
        for i in range(num_blobs):
            if use_brackets:
                w = x1_arr[i] - x0_arr[i]
                h = y1_arr[i] - y0_arr[i]
                bracket_w = int(w * bracket_length)
                bracket_h = int(h * bracket_length)
                
                cv2.line(out_img, (x0_arr[i], y0_arr[i]), (x0_arr[i] + bracket_w, y0_arr[i]), outline_col_norm, blob_thickness)
                cv2.line(out_img, (x0_arr[i], y0_arr[i]), (x0_arr[i], y0_arr[i] + bracket_h), outline_col_norm, blob_thickness)
                cv2.line(out_img, (x1_arr[i], y0_arr[i]), (x1_arr[i] - bracket_w, y0_arr[i]), outline_col_norm, blob_thickness)
                cv2.line(out_img, (x1_arr[i], y0_arr[i]), (x1_arr[i], y0_arr[i] + bracket_h), outline_col_norm, blob_thickness)
                cv2.line(out_img, (x0_arr[i], y1_arr[i]), (x0_arr[i] + bracket_w, y1_arr[i]), outline_col_norm, blob_thickness)
                cv2.line(out_img, (x0_arr[i], y1_arr[i]), (x0_arr[i], y1_arr[i] - bracket_h), outline_col_norm, blob_thickness)
                cv2.line(out_img, (x1_arr[i], y1_arr[i]), (x1_arr[i] - bracket_w, y1_arr[i]), outline_col_norm, blob_thickness)
                cv2.line(out_img, (x1_arr[i], y1_arr[i]), (x1_arr[i], y1_arr[i] - bracket_h), outline_col_norm, blob_thickness)
            else:
                cv2.rectangle(out_img, (x0_arr[i], y0_arr[i]), (x1_arr[i], y1_arr[i]), 
                            outline_col_norm, blob_thickness)
            
            cv2.rectangle(blobs_img, (x0_arr[i], y0_arr[i]), (x1_arr[i], y1_arr[i]), 
                        white, -1)
        
        # Text labels
        if show_ids or show_leaders:
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(num_blobs):
                text_parts = []
                if show_ids and show_xy:
                    x_norm = centers_x[i] / out_w
                    y_norm = centers_y[i] / out_h
                    text_parts.append(f"x:{x_norm:.2f} y:{y_norm:.2f}")
                elif show_ids:
                    blob_id = _frame_skip_cache.get('blob_ids', [i])[i] if i < len(_frame_skip_cache.get('blob_ids', [])) else i
                    text_parts.append(f"ID {blob_id}")
                
                if text_parts:
                    text = " ".join(text_parts)
                    h_box = y1_arr[i] - y0_arr[i]
                    font_scale = np.clip(h_box / 100, 0.25, 0.4)
                    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
                    tx = x0_arr[i]
                    ty = max(y0_arr[i] - 2, th)
                    
                    if show_leaders:
                        cv2.line(out_img, (centers_x[i], centers_y[i]), (tx, ty + th//2), outline_col_norm, 1, cv2.LINE_4)
                    
                    cv2.putText(out_img, text, (tx, ty), font, font_scale, white, 1, cv2.LINE_4)
        
        # Metrics overlay
        if show_metrics:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            line_height = 20
            padding = 10
            y_pos = padding + line_height
            
            cv2.putText(out_img, "TRACKING DATA", (padding, y_pos), font, font_scale, outline_col_norm, 1, cv2.LINE_4)
            y_pos += int(line_height * 1.5)
            
            for i in range(num_blobs):
                blob_id = _frame_skip_cache.get('blob_ids', [i])[i] if i < len(_frame_skip_cache.get('blob_ids', [])) else i
                x_norm = centers_x[i] / out_w
                y_norm = centers_y[i] / out_h
                size_val = y1_arr[i] - y0_arr[i]
                conf = _frame_skip_cache.get('blob_confidence', {}).get(blob_id, 0)
                data_text = f"ID:{blob_id} X:{x_norm:.2f} Y:{y_norm:.2f} SZ:{size_val} CONF:{conf}"
                cv2.putText(out_img, data_text, (padding, y_pos), font, font_scale * 0.9, white, 1, cv2.LINE_4)
                y_pos += line_height
        
        # Connection lines
        if draw_connections and num_blobs > 1:
            avg_size = np.mean(y1_arr - y0_arr)
            connection_distance = avg_size * 3
            connection_thickness = max(1, int(blob_thickness * 0.5))
            
            for i in range(num_blobs):
                for j in range(i + 1, num_blobs):
                    dx = centers_x[j] - centers_x[i]
                    dy = centers_y[j] - centers_y[i]
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist <= connection_distance:
                        pt1 = (centers_x[i], centers_y[i])
                        pt2 = (centers_x[j], centers_y[j])
                        
                        if use_dotted:
                            draw_dotted_line(out_img, pt1, pt2, outline_col_norm, connection_thickness)
                        else:
                            cv2.line(out_img, pt1, pt2, outline_col_norm, connection_thickness, cv2.LINE_4)
    
    # Grid overlay
    if show_grid:
        grid_col = (outline_col[0]/255.0 * 0.3, outline_col[1]/255.0 * 0.3, outline_col[2]/255.0 * 0.3, 0.5)
        x = 0
        while x < out_w:
            cv2.line(out_img, (int(x), 0), (int(x), out_h), grid_col, 1, cv2.LINE_4)
            x += grid_spacing
        y = 0
        while y < out_h:
            cv2.line(out_img, (0, int(y)), (out_w, int(y)), grid_col, 1, cv2.LINE_4)
            y += grid_spacing

    # Draw trails
    if draw_trails and centers and len(centers) >= 2:
        trail_pts = centers
        
        if len(trail_pts) >= 4:
            trail_pts = interpolate_catmull_rom_cached(trail_pts, resolution=line_smoothness)

        if len(trail_pts) >= 2:
            trail_array = np.array(trail_pts, dtype=np.float32)
            diffs = trail_array[1:] - trail_array[:-1]
            segment_lengths = np.sqrt(np.sum(diffs * diffs, axis=1))
            full_length = np.sum(segment_lengths)
            visible_length = full_length * np.clip(max_line_length_norm, 0.0, 1.0)

            if visible_length < full_length:
                cumsum_rev = np.cumsum(segment_lengths[::-1])
                cutoff_idx = np.searchsorted(cumsum_rev, visible_length)
                
                if cutoff_idx < len(trail_pts) - 1:
                    start_idx = max(0, len(trail_pts) - cutoff_idx - 2)
                    trimmed = trail_pts[start_idx:]
                else:
                    trimmed = trail_pts
            else:
                trimmed = trail_pts

            if len(trimmed) >= 2:
                if use_dotted:
                    for i in range(len(trimmed) - 1):
                        pt1 = (int(trimmed[i][0]), int(trimmed[i][1]))
                        pt2 = (int(trimmed[i+1][0]), int(trimmed[i+1][1]))
                        draw_dotted_line(out_img, pt1, pt2, trail_col_norm, 1)
                else:
                    pts = np.array(trimmed, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(out_img, [pts], False, trail_col_norm, 1, cv2.LINE_4)

    out_img = cv2.flip(out_img, 0)
    blobs_img = cv2.flip(blobs_img, 0)

    scriptOp.copyNumpyArray(out_img)
    comp.store('blobs_image', blobs_img)
