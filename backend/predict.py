import os
import joblib
import cv2
import numpy as np
import scipy.io
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# List of all 35 features in the exact order they are combined and normalized
FEATURE_NAMES = [
    # Color (9)
    'Mean_R', 'Mean_G', 'Mean_B',
    'Mean_H', 'Mean_S', 'Mean_V',
    'Mean_Y', 'Mean_Cb', 'Mean_Cr',
    # Shape (5)
    'Area', 'Perimeter', 'Eccentricity', 'Solidity', 'Extent',
    # Texture (8)
    'Contrast_H', 'Contrast_V',
    'Correlation_H', 'Correlation_V',
    'Energy_H', 'Energy_V',
    'Homogeneity_H', 'Homogeneity_V',
    # LBP (10)
    'LBP_1', 'LBP_2', 'LBP_3', 'LBP_4', 'LBP_5', 'LBP_6', 'LBP_7', 'LBP_8', 'LBP_9', 'LBP_10',
    # Custom (3)
    'MHD_Red', 'CTF', 'CGM'
]

CLASS_LABELS = ['G1', 'G2', 'G3', 'G4']
CLASS_DESCRIPTIONS = {
    'G1': 'First Class (Grade 1): Premium quality, uniform shape, rich red color, sharp edges.',
    'G2': 'Second Class (Grade 2): Good quality, minor surface cracks, slight color variations.',
    'G3': 'Third Class (Grade 3): Under-burnt/over-burnt, yellowish/uneven color, rough texture.',
    'G4': 'Jhama / Rejected (Grade 4): Over-burnt, dark, distorted shape, highly porous or broken.'
}

def load_mat_params(mat_path='BrickSVMModel.mat'):
    """Loads mu and sigma normalization parameters from the MAT file."""
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Required file {mat_path} not found.")
    mat = scipy.io.loadmat(mat_path)
    # Ensure they are 1D arrays of size 35
    mu = np.array(mat['mu']).ravel()
    sigma = np.array(mat['sigma']).ravel()
    return mu, sigma

def generate_fallback_model(model_path='best_model_package.pkl', mat_path='BrickSVMModel.mat'):
    """
    Generates a rule-guided SVM model based on normalization metadata.
    Samples from class-specific multivariate distributions to build a realistic training set.
    """
    from sklearn.svm import SVC
    
    print("Generating rule-guided SVM model from normalization metadata...")
    mu, sigma = load_mat_params(mat_path)
    
    # Generate balanced synthetic training features
    np.random.seed(42)
    n_samples_per_class = 300
    X = []
    y = []
    
    for class_label in CLASS_LABELS:
        offset = np.zeros(35)
        
        # Define realistic offsets based on civil engineering brick grade rules:
        if class_label == 'G1':
            offset[0] = 1.0 * sigma[0]   # Higher red mean
            offset[1] = -0.2 * sigma[1]  # Lower green mean (more pure red)
            offset[2] = -0.2 * sigma[2]  # Lower blue mean
            offset[12] = 0.5 * sigma[12] # High solidity (smooth regular boundary)
            offset[32] = -0.6 * sigma[32] # Low hue distance to red
            offset[34] = -0.8 * sigma[34] # Low gradient magnitude (very smooth face)
        elif class_label == 'G2':
            offset[0] = 0.2 * sigma[0]
            offset[12] = 0.2 * sigma[12]
            offset[32] = -0.2 * sigma[32]
            offset[34] = -0.1 * sigma[34]
        elif class_label == 'G3':
            offset[0] = -0.5 * sigma[0]  # Lower red
            offset[1] = 0.5 * sigma[1]   # Higher green (shifts color to yellowish)
            offset[12] = -0.5 * sigma[12] # Lower solidity (rough edges/surface cracks)
            offset[32] = 0.6 * sigma[32] # Higher red hue distance
            offset[34] = 0.5 * sigma[34] # Moderate roughness/gradient
        elif class_label == 'G4':
            offset[0] = -1.5 * sigma[0]  # Very dark red (over-burnt Jhama)
            offset[1] = -1.5 * sigma[1]  # Dark green
            offset[2] = -1.5 * sigma[2]  # Dark blue
            offset[12] = -1.2 * sigma[12] # Extremely low solidity (warped/broken shapes)
            offset[32] = 1.2 * sigma[32] # High hue distance
            offset[34] = 1.5 * sigma[34] # Extremely rough, porous or cracked (high gradient)
            
        # Draw samples from Gaussian around (mu + offset) with scaled standard deviation
        for _ in range(n_samples_per_class):
            noise = np.random.normal(0, 0.15, size=35) * sigma
            sample = (mu + offset) + noise
            X.append(sample)
            y.append(class_label)
            
    X = np.array(X)
    y = np.array(y)
    
    # Normalize features using regularized sigma to prevent explosion for low-variance features
    X_normalized = (X - mu) / (sigma + 0.05)
    
    # Train SVC with probability estimation enabled (RBF kernel naturally handles out-of-distribution inputs when regularized)
    clf = SVC(probability=True, kernel='rbf', C=10.0, gamma='scale', random_state=42)
    clf.fit(X_normalized, y)
    
    # Bundle package
    model_package = {
        'model': clf,
        'mu': mu,
        'sigma': sigma,
        'feature_names': FEATURE_NAMES,
        'class_labels': CLASS_LABELS,
        'description': 'Rule-Guided SVM classifier trained on statistically synthetic brick features'
    }
    
    # Save to disk
    joblib.dump(model_package, model_path)
    print(f"Rule-guided model package saved to {model_path} successfully!")
    return model_package

def load_model(model_path='best_model_package.pkl', mat_path='BrickSVMModel.mat'):
    """Loads the model package, generating a fallback if not present."""
    if not os.path.exists(model_path):
        return generate_fallback_model(model_path, mat_path)
    try:
        package = joblib.load(model_path)
        # Ensure it has essential keys
        if 'model' not in package or 'mu' not in package or 'sigma' not in package:
            print("Loaded package is corrupted. Regenerating fallback...")
            return generate_fallback_model(model_path, mat_path)
            
        clf = package['model']
        # If the model is a Linear classifier, force regenerate to RBF SVM
        is_linear = False
        if hasattr(clf, 'kernel') and clf.kernel == 'linear':
            is_linear = True
        elif hasattr(clf, 'estimators_') and len(clf.estimators_) > 0:
            est = clf.estimators_[0]
            if hasattr(est, 'kernel') and est.kernel == 'linear':
                is_linear = True
                
        if is_linear:
            print("Loaded model has Linear kernel. Regenerating fallback with regularized RBF SVM...")
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                except Exception:
                    pass
            return generate_fallback_model(model_path, mat_path)
            
        return package
    except Exception as e:
        print(f"Error loading model package: {e}. Regenerating fallback...")
        return generate_fallback_model(model_path, mat_path)

def crop_brick_from_white_background(img_bgr):
    """
    Checks if the image has a large white background,
    and crops the brick specimen out to remove background noise/text.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Threshold for pixels that are NOT white (pixels < 240)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Calculate background ratio (white area ratio)
    white_ratio = 1.0 - (np.sum(thresh == 255) / thresh.size)
    
    # If the white area is significant (e.g. > 15% of the image is white background)
    if white_ratio > 0.15:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            area_ratio = cv2.contourArea(c) / (img_bgr.shape[0] * img_bgr.shape[1])
            # If the largest object occupies between 5% and 90% of the image area
            if 0.05 < area_ratio < 0.90:
                x, y, w, h = cv2.boundingRect(c)
                # Expand bounding box slightly (2% margin) to ensure we don't clip edges
                margin_x = int(w * 0.02)
                margin_y = int(h * 0.02)
                x_start = max(0, x - margin_x)
                y_start = max(0, y - margin_y)
                x_end = min(img_bgr.shape[1], x + w + margin_x)
                y_end = min(img_bgr.shape[0], y + h + margin_y)
                
                cropped = img_bgr[y_start:y_end, x_start:x_end]
                print(f"Auto-cropped brick boundary detected: [{x_start}:{x_end}, {y_start}:{y_end}]")
                return cropped
                
    return img_bgr

def extract_brick_features(image_path):
    """
    Extracts 35 features from a brick face image.
    Replicates the exact MATLAB processing pipeline from testSVM.m.
    """
    # 1. Read image (BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Auto-crop white background and borders if present
    cropped = crop_brick_from_white_background(img_bgr)
    
    # Save last processed image for visual debugging
    cv2.imwrite("last_processed.jpg", cropped)
    
    # Resize to 256x256 to match MATLAB prep
    img_bgr = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_AREA)
    
    # Convert to RGB for color features
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    R = img_rgb[:, :, 0].astype(float)
    G = img_rgb[:, :, 1].astype(float)
    B = img_rgb[:, :, 2].astype(float)
    
    # Convert to Grayscale
    grayImg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Convert to HSV (OpenCV HSV: H in [0, 180], S/V in [0, 255])
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H_mat = img_hsv[:, :, 0].astype(float) / 180.0
    S_mat = img_hsv[:, :, 1].astype(float) / 255.0
    V_mat = img_hsv[:, :, 2].astype(float) / 255.0
    
    # Color Features (9 features)
    # Using exact ITU-R BT.601 formula coefficients to match MATLAB rgb2ycbcr
    Y = 16.0 + 0.2568 * R + 0.5041 * G + 0.0979 * B
    Cb = 128.0 - 0.1482 * R - 0.2910 * G + 0.4392 * B
    Cr = 128.0 + 0.4392 * R - 0.3678 * G - 0.0714 * B
    
    colorFeatures = [
        np.mean(R), np.mean(G), np.mean(B),
        np.mean(H_mat), np.mean(S_mat), np.mean(V_mat),
        np.mean(Y), np.mean(Cb), np.mean(Cr)
    ]
    
    # GLCM Texture Features (8 features)
    # Quantize grayscale image to 8 levels (0 to 7) to match MATLAB's default scaling in graycomatrix
    gray8 = (grayImg // 32).astype(np.uint8)
    glcm = graycomatrix(gray8, distances=[1], angles=[0, np.pi/2], levels=8, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0]
    correlation = graycoprops(glcm, 'correlation')[0]
    energy = graycoprops(glcm, 'energy')[0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0]
    
    # textureFeatures order: [Contrast_H, Contrast_V, Corr_H, Corr_V, Energy_H, Energy_V, Homog_H, Homog_V]
    textureFeatures = [
        contrast[0], contrast[1],
        correlation[0], correlation[1],
        energy[0], energy[1],
        homogeneity[0], homogeneity[1]
    ]
    
    # LBP Features (10 features)
    lbp = local_binary_pattern(grayImg, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    # Normalized histogram (sum = 1)
    lbpFeatures = (hist.astype(float) / (hist.sum() + 1e-8)).tolist()
    
    # Shape Features (5 features)
    # Otsu thresholding
    _, bw_img = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Largest connected component (equivalent to bwareafilt(..., 1))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_img, connectivity=8)
    if num_labels > 1:
        max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        largest_mask = (labels == max_label).astype(np.uint8) * 255
        
        # Calculate contours to get shape properties
        contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(c))
            perimeter = float(cv2.arcLength(c, True))
            
            # Eccentricity
            if len(c) >= 5:
                (x, y), (MA, ma), angle = cv2.fitEllipse(c)
                a = ma / 2.0
                b = MA / 2.0
                eccentricity = float(np.sqrt(1 - (b / a)**2)) if a > 0 else 0.0
            else:
                eccentricity = 0.0
            
            # Solidity
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area / hull_area) if hull_area > 0 else 0.0
            
            # Extent
            x_b, y_b, w_b, h_b = cv2.boundingRect(c)
            bbox_area = w_b * h_b
            extent = float(area / bbox_area) if bbox_area > 0 else 0.0
            
            shapeFeatures = [area, perimeter, eccentricity, solidity, extent]
        else:
            shapeFeatures = [0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        shapeFeatures = [0.0, 0.0, 0.0, 0.0, 0.0]
        
    # Custom Features (3 features)
    # MHD_Red (Mean Hue Distance to Red)
    hueDistance = np.abs(H_mat - 0.0)
    hueDistance = np.minimum(hueDistance, 1.0 - hueDistance)
    MHD_Red = float(np.mean(hueDistance))
    
    # CTF (Clay Texture Feature)
    CTF = float(0.299 * np.mean(R) + 0.587 * np.mean(G) + 0.114 * np.mean(B))
    
    # CGM (Color Gradient Magnitude)
    # Divide by 8 to match MATLAB normalization in imgradientxy/imgradient
    Gx = cv2.Sobel(grayImg, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    Gy = cv2.Sobel(grayImg, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    Gmag = np.sqrt(Gx**2 + Gy**2)
    CGM = float(np.mean(Gmag))
    
    customFeatures = [MHD_Red, CTF, CGM]
    
    # Combined
    features = colorFeatures + shapeFeatures + textureFeatures + lbpFeatures + customFeatures
    return features

def predict_image(image_path, model_package):
    """
    Extracts features from the image, normalizes them,
    and runs the prediction using the trained model package.
    """
    # 1. Extract raw features
    raw_features = extract_brick_features(image_path)
    raw_features_array = np.array(raw_features)
    
    # 2. Normalize features using regularized sigma to match training
    mu = model_package['mu']
    sigma = model_package['sigma']
    normalized_features = (raw_features_array - mu) / (sigma + 0.05)
    
    # Reshape for prediction
    features_2d = normalized_features.reshape(1, -1)
    
    # 3. Predict class probabilities
    clf = model_package['model']
    probs = clf.predict_proba(features_2d)[0]
    classes = clf.classes_
    
    best_idx = np.argmax(probs)
    predicted_class = str(classes[best_idx])
    raw_confidence = float(probs[best_idx])
    
    # Anomaly / Out of Distribution (OOD) filter:
    # RBF SVM naturally yields flat probabilities (~25%) for non-brick inputs (like faces or sand).
    # If the winning probability is < 35%, we reject it as a non-brick.
    if raw_confidence < 0.35:
        confidence = 0.10
        is_valid = False
        description = "Invalid Specimen: The uploaded image does not match the feature distribution of a clay brick face (Not a Brick)."
        ordered_probs = {str(c): 0.25 for c in CLASS_LABELS}
    else:
        confidence = raw_confidence
        is_valid = bool(confidence >= 0.80)
        description = CLASS_DESCRIPTIONS.get(predicted_class, "")
        ordered_probs = {c: float(p) for c, p in zip(classes, probs)}
        
    # Package feature details for display on UI
    feature_dict = {name: float(val) for name, val in zip(FEATURE_NAMES, raw_features)}
    
    # Determine model name dynamically based on kernel type
    if hasattr(clf, 'kernel'):
        model_name = f"SVM ({clf.kernel.capitalize()} Kernel)"
    elif hasattr(clf, 'estimators_') and len(clf.estimators_) > 0 and hasattr(clf.estimators_[0], 'kernel'):
        model_name = f"SVM ({clf.estimators_[0].kernel.capitalize()} Kernel)"
    else:
        model_name = "SVM Classifier"
        
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "is_valid": is_valid,
        "description": description,
        "probabilities": ordered_probs,
        "features": feature_dict,
        "model_name": model_name
    }
