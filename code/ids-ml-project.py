# Intrusion Detection System using Machine Learning
# Author: Claude
# Date: April 7, 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# 1. Data Loading and Exploration
# =====================================================

def load_data():
    """
    Load the NSL-KDD dataset.
    Returns the training and testing dataframes.
    """
    print("Loading NSL-KDD dataset...")
    
    # Define column names for the dataset
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_type", "difficulty"]
    
    # Load the training dataset
    train_data = pd.read_csv("KDDTrain+.txt", header=None, names=col_names)
    
    # Load the testing dataset
    test_data = pd.read_csv("KDDTest+.txt", header=None, names=col_names)
    
    # Display basic information
    print(f"Training set shape: {train_data.shape}")
    print(f"Testing set shape: {test_data.shape}")
    
    return train_data, test_data

def explore_dataset(data):
    """
    Perform basic exploratory data analysis on the dataset.
    """
    print("\nDataset Overview:")
    print(data.head())
    
    print("\nData Types:")
    print(data.dtypes)
    
    print("\nDetecting missing values:")
    print(data.isnull().sum().sum())
    
    # Attack type distribution
    print("\nAttack Type Distribution:")
    print(data['attack_type'].value_counts())
    
    # Map attack types to their respective categories
    attack_cat = {
        'normal': 'normal',
        'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 
        'smurf': 'dos', 'teardrop': 'dos', 'apache2': 'dos', 'udpstorm': 'dos', 
        'processtable': 'dos', 'worm': 'dos', 'mailbomb': 'dos',
        'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 
        'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
        'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 
        'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l', 
        'warezmaster': 'r2l', 'sendmail': 'r2l', 'named': 'r2l', 'snmpgetattack': 'r2l',
        'snmpguess': 'r2l', 'xlock': 'r2l', 'xsnoop': 'r2l', 'httptunnel': 'r2l',
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 
        'rootkit': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r'
    }
    
    # Map attack types to broader categories
    data['attack_category'] = data['attack_type'].map(lambda x: attack_cat.get(x, 'unknown'))
    
    print("\nAttack Category Distribution:")
    print(data['attack_category'].value_counts())
    
    return data

def visualize_data(data):
    """
    Create visualizations for better understanding of the dataset.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot attack categories distribution
    plt.subplot(1, 2, 1)
    sns.countplot(y='attack_category', data=data, order=data['attack_category'].value_counts().index)
    plt.title('Attack Categories')
    plt.xlabel('Count')
    plt.ylabel('Category')
    
    # Plot protocol type distribution
    plt.subplot(1, 2, 2)
    sns.countplot(y='protocol_type', data=data)
    plt.title('Protocol Types')
    plt.xlabel('Count')
    plt.ylabel('Protocol')
    
    plt.tight_layout()
    plt.savefig('attack_distributions.png')
    print("Saved visualization to 'attack_distributions.png'")
    
    # Visualize correlations between numerical features
    plt.figure(figsize=(20, 16))
    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, square=True)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('feature_correlations.png')
    print("Saved correlation matrix to 'feature_correlations.png'")

# =====================================================
# 2. Data Preprocessing
# =====================================================

def preprocess_data(train_data, test_data):
    """
    Preprocess the dataset by:
    - Converting categorical features
    - Scaling numerical features
    - Feature selection
    """
    print("\nPreprocessing data...")
    
    # Make copies to avoid modifying originals
    train = train_data.copy()
    test = test_data.copy()
    
    # Extract target variables
    y_train = train['attack_category']
    y_test = test['attack_category']
    
    # Drop unnecessary columns
    cols_to_drop = ['attack_type', 'attack_category', 'difficulty']
    X_train = train.drop(cols_to_drop, axis=1)
    X_test = test.drop(cols_to_drop, axis=1)
    
    # Identify categorical columns
    categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
    print(f"Categorical columns: {categorical_cols}")
    
    # Convert categorical features using one-hot encoding
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    
    # Make sure both dataframes have the same columns
    # Add missing columns to the test set
    missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
    for col in missing_cols:
        X_test_encoded[col] = 0
        
    # Ensure columns are in the same order
    X_test_encoded = X_test_encoded[X_train_encoded.columns]
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)
    
    print(f"Preprocessed training data shape: {X_train_scaled.shape}")
    print(f"Preprocessed testing data shape: {X_test_scaled.shape}")
    
    # Convert target to numeric using LabelEncoder
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Track the label mapping for later reference
    label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print(f"Target class mapping: {label_mapping}")
    
    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, le.classes_

def select_features(X_train, X_test, y_train, k=20):
    """
    Perform feature selection to reduce dimensionality.
    """
    print(f"\nSelecting top {k} features...")
    
    # Select K best features
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    print(f"Selected {len(selected_indices)} features")
    
    return X_train_selected, X_test_selected

# =====================================================
# 3. Model Building and Evaluation
# =====================================================

def build_models(X_train, y_train):
    """
    Build machine learning models for intrusion detection.
    """
    print("\nBuilding machine learning models...")
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    # Train models
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def evaluate_models(models, X_test, y_test, class_names):
    """
    Evaluate models and print performance metrics.
    """
    print("\nEvaluating models...")
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name} Results:")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=class_names)
        print("Classification Report:")
        print(report)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
        print(f"Saved confusion matrix to 'confusion_matrix_{name.replace(' ', '_').lower()}.png'")
        
        results[name] = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm
        }
    
    return results

# =====================================================
# 4. Main Function
# =====================================================

def main():
    """
    Main function to orchestrate the IDS project.
    """
    print("=" * 50)
    print("Intrusion Detection System using Machine Learning")
    print("=" * 50)
    
    # Step 1: Load the dataset
    try:
        train_data, test_data = load_data()
    except FileNotFoundError:
        print("Dataset files not found. Please download the NSL-KDD dataset and place the files in the current directory.")
        print("Download from: https://www.unb.ca/cic/datasets/nsl.html")
        print("Required files: KDDTrain+.txt and KDDTest+.txt")
        return
    
    # Step 2: Explore and visualize the dataset
    train_data = explore_dataset(train_data)
    visualize_data(train_data)
    
    # Step 3: Preprocess the data
    X_train, X_test, y_train, y_test, class_names = preprocess_data(train_data, test_data)
    
    # Step 4: Feature selection
    X_train_selected, X_test_selected = select_features(X_train, X_test, y_train)
    
    # Step 5: Build machine learning models
    models = build_models(X_train_selected, y_train)
    
    # Step 6: Evaluate models
    results = evaluate_models(models, X_test_selected, y_test, class_names)
    
    print("\nIDS project completed successfully!")
    
    # Return best model and results
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\nBest performing model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")

if __name__ == "__main__":
    main()
