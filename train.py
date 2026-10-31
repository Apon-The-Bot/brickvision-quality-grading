import os
import argparse
import time
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import predict

def main():
    parser = argparse.ArgumentParser(description="Train a Brick Grade Classification SVM Model.")
    parser.add_argument("--dataset_dir", type=str, required=True, 
                        help="Path to the dataset directory containing subfolders G1, G2, G3, G4.")
    parser.add_argument("--output_model", type=str, default="best_model_package.pkl",
                        help="Path to save the trained model package (default: best_model_package.pkl).")
    parser.add_argument("--test_size", type=float, default=0.20,
                        help="Ratio of testing data (default: 0.20).")
    args = parser.parse_args()

    dataset_path = args.dataset_dir
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory {dataset_path} does not exist.")
        return

    print(f"Loading brick dataset from: {dataset_path}...")
    X = []
    y = []
    
    classes = predict.CLASS_LABELS
    for class_label in classes:
        class_dir = os.path.join(dataset_path, class_label)
        if not os.path.exists(class_dir):
            print(f"Warning: Subfolder for class {class_label} not found at {class_dir}. Skipping.")
            continue
            
        file_list = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        print(f"Processing class {class_label}: found {len(file_list)} images...")
        
        for file_name in file_list:
            img_path = os.path.join(class_dir, file_name)
            try:
                # Extract 35 features
                feats = predict.extract_brick_features(img_path)
                X.append(feats)
                y.append(class_label)
            except Exception as e:
                print(f"  Error processing image {img_path}: {e}. Skipping.")

    if len(X) == 0:
        print("Error: No images were successfully processed. Training aborted.")
        return

    X = np.array(X)
    y = np.array(y)
    print(f"\nSuccessfully loaded {X.shape[0]} total samples.")
    print("Feature matrix shape:", X.shape)
    
    # Calculate normalization parameters (mu and sigma) on the training set
    print("Computing normalization statistics...")
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    # Prevent divide-by-zero for constant features
    sigma[sigma == 0.0] = 1.0
    
    # Normalize features
    X_normalized = (X - mu) / sigma
    
    # Train-test split
    print(f"Splitting dataset (Holdout={args.test_size * 100:.0f}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    # Setup multiclass SVM classifier using One-vs-Rest coding
    print("Training SVM Model with RBF Kernel (One-vs-Rest)...")
    start_time = time.time()
    
    # base estimator
    base_svc = SVC(probability=True, kernel='rbf', C=10.0, gamma='scale', random_state=42)
    # One-vs-Rest wrapper to match MATLAB's ECOC one-vs-all configuration
    ovr_classifier = OneVsRestClassifier(base_svc)
    ovr_classifier.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    # Evaluate model
    y_pred = ovr_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=classes))
    
    # Save the trained model package
    model_package = {
        'model': ovr_classifier,
        'mu': mu,
        'sigma': sigma,
        'feature_names': predict.FEATURE_NAMES,
        'class_labels': classes,
        'description': f'Brick SVM (One-vs-Rest, RBF Kernel) trained on {X.shape[0]} samples.'
    }
    
    joblib.dump(model_package, args.output_model)
    print(f"\nModel package saved successfully to {args.output_model}!")

if __name__ == '__main__':
    main()
