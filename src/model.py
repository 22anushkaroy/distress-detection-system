import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import joblib

class DistressModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_count = 18
        self.class_names = {
            1: "Walking",
            2: "Walking Upstairs",
            3: "Walking Downstairs",
            4: "Sitting",
            5: "Standing",
            6: "Laying",
            7: "Anomaly",
            8: "Fall",
            9: "Panic Run",
            10: "Struggling",
            11: "Freeze"
        }

    def extract_features_from_raw(self, ax, ay, az):
        """Extract 18 features from raw accelerometer data (6 per axis)"""
        features = []
        for axis_data in [ax, ay, az]:
            axis_data = np.array(axis_data, dtype=float)
            features.append(np.mean(axis_data))
            features.append(np.std(axis_data))
            features.append(np.min(axis_data))
            features.append(np.max(axis_data))
            features.append(np.max(axis_data) - np.min(axis_data))
            features.append(np.sqrt(np.mean(axis_data**2)))
        return features

    def read_sensor_csv(self, filepath):
        """
        Read Physics Toolbox Sensor Suite CSV files
        Handles:
        - Comment lines starting with #
        - Column names with units like 'ax (m/s^2)'
        - Extra columns like aT
        """
        # Count how many comment lines to skip
        skip_rows = 0
        with open(filepath, 'r') as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith('#'):
                    skip_rows += 1
                else:
                    break

        # Read CSV skipping comment lines
        df = pd.read_csv(filepath, skiprows=skip_rows)

        # Print columns for debugging
        print(f"   Columns found: {list(df.columns)}")

        # Find accelerometer columns by checking for ax, ay, az in column names
        ax_col = None
        ay_col = None
        az_col = None

        for col in df.columns:
            col_lower = col.lower().strip()

            # Match 'ax' but not 'max' or 'aT'
            if ax_col is None:
                if col_lower.startswith('ax') or 'ax (' in col_lower:
                    ax_col = col

            if ay_col is None:
                if col_lower.startswith('ay') or 'ay (' in col_lower:
                    ay_col = col

            if az_col is None:
                if col_lower.startswith('az') or 'az (' in col_lower:
                    az_col = col

        # If still not found, try by column position
        # Expected: time, ax, ay, az, aT (5 columns)
        if ax_col is None or ay_col is None or az_col is None:
            cols = list(df.columns)
            if len(cols) >= 4:
                # Skip first column (time), take next 3
                ax_col = cols[1]
                ay_col = cols[2]
                az_col = cols[3]
                print(f"   Using columns by position: {ax_col}, {ay_col}, {az_col}")
            else:
                return None, None, None, "Not enough columns"

        print(f"   Using: ax='{ax_col}', ay='{ay_col}', az='{az_col}'")

        # Extract data and convert to numeric
        ax = pd.to_numeric(df[ax_col], errors='coerce').dropna().values
        ay = pd.to_numeric(df[ay_col], errors='coerce').dropna().values
        az = pd.to_numeric(df[az_col], errors='coerce').dropna().values

        # Make sure all arrays are same length
        min_len = min(len(ax), len(ay), len(az))
        ax = ax[:min_len]
        ay = ay[:min_len]
        az = az[:min_len]

        return ax, ay, az, None

    def load_my_sensor_data(self, folder_path):
        """Load your collected CSV sensor data"""
        print("\n📱 Loading your collected sensor data...")

        my_data = []
        my_labels = []

        file_class_map = {
            'normal': 1,
            'fall': 8,
            'panic': 9,
            'struggle': 10,
            'freeze': 11
        }

        if not os.path.exists(folder_path):
            print(f"⚠️  Folder not found: {folder_path}")
            return np.array([]), np.array([])

        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        if len(files) == 0:
            print(f"⚠️  No CSV files found in {folder_path}")
            return np.array([]), np.array([])

        for filename in files:
            filepath = os.path.join(folder_path, filename)

            # Determine class from filename
            class_label = None
            for prefix, label in file_class_map.items():
                if filename.lower().startswith(prefix):
                    class_label = label
                    break

            if class_label is None:
                print(f"⚠️  Skipping unknown file: {filename}")
                continue

            try:
                # Read CSV using our custom reader
                ax, ay, az, error = self.read_sensor_csv(filepath)

                if error:
                    print(f"⚠️  Error in {filename}: {error}")
                    continue

                if len(ax) < 10:
                    print(f"⚠️  Too few data points in {filename}: {len(ax)}")
                    continue

                print(f"   Total data points: {len(ax)}")

                # Sliding window to create multiple samples
                window_size = 128
                step_size = 64
                samples_created = 0

                if len(ax) < window_size:
                    # Pad short recordings
                    ax_padded = np.pad(ax, (0, window_size - len(ax)), mode='edge')
                    ay_padded = np.pad(ay, (0, window_size - len(ay)), mode='edge')
                    az_padded = np.pad(az, (0, window_size - len(az)), mode='edge')

                    features = self.extract_features_from_raw(ax_padded, ay_padded, az_padded)
                    my_data.append(features)
                    my_labels.append(class_label)
                    samples_created = 1
                else:
                    for i in range(0, len(ax) - window_size, step_size):
                        ax_window = ax[i:i + window_size]
                        ay_window = ay[i:i + window_size]
                        az_window = az[i:i + window_size]

                        features = self.extract_features_from_raw(ax_window, ay_window, az_window)
                        my_data.append(features)
                        my_labels.append(class_label)
                        samples_created += 1

                print(f"✓ Loaded {filename} → Class {class_label} ({self.class_names[class_label]}) → {samples_created} samples")

            except Exception as e:
                print(f"⚠️  Error loading {filename}: {e}")
                continue

        if len(my_data) > 0:
            print(f"✓ Total samples from your data: {len(my_data)}")
            return np.array(my_data), np.array(my_labels)
        else:
            print("⚠️  No samples could be extracted from your data")
            return np.array([]), np.array([])

    def load_uci_har_data(self, train_folder):
        """Load UCI HAR dataset and reduce to 18 features"""
        print("\n📊 Loading UCI HAR dataset...")

        X_path = os.path.join(train_folder, 'X_train.txt')
        y_path = os.path.join(train_folder, 'y_train.txt')

        if not os.path.exists(X_path) or not os.path.exists(y_path):
            print(f"⚠️  UCI HAR dataset not found in {train_folder}")
            return np.array([]), np.array([])

        try:
            X = np.loadtxt(X_path)
            y = np.loadtxt(y_path)

            # Take first 18 features
            X_reduced = X[:, :18]

            print(f"✓ Loaded {len(X_reduced)} samples from UCI HAR")
            return X_reduced, y

        except Exception as e:
            print(f"⚠️  Error loading UCI HAR: {e}")
            return np.array([]), np.array([])

    def generate_synthetic_distress(self, n_samples=200):
        """Generate synthetic distress patterns"""
        print("\n🔬 Generating synthetic distress data...")

        synthetic_data = []
        synthetic_labels = []

        for _ in range(n_samples):
            # Class 7: Random anomaly
            features = np.random.randn(18) * 5
            synthetic_data.append(features)
            synthetic_labels.append(7)

            # Class 8: Fall (big spike then stillness)
            features = [20, 15, -20, 25, 45, 22,
                        -18, 12, -15, 20, 35, 19,
                        -30, 8, -35, -5, 30, 28]
            features = np.array(features) + np.random.randn(18) * 2
            synthetic_data.append(features)
            synthetic_labels.append(8)

            # Class 9: Panic run (chaotic high movement)
            features = np.random.randn(18) * 8 + 3
            synthetic_data.append(features)
            synthetic_labels.append(9)

            # Class 10: Struggling (jerky movements)
            features = []
            for _ in range(3):
                features.extend([
                    np.random.uniform(-15, 15),
                    np.random.uniform(8, 20),
                    np.random.uniform(-25, -10),
                    np.random.uniform(10, 25),
                    np.random.uniform(30, 50),
                    np.random.uniform(10, 20)
                ])
            features = np.array(features[:18])
            synthetic_data.append(features)
            synthetic_labels.append(10)

            # Class 11: Freeze (very low movement)
            features = np.random.randn(18) * 0.5
            synthetic_data.append(features)
            synthetic_labels.append(11)

        print(f"✓ Generated {len(synthetic_data)} synthetic samples")
        return np.array(synthetic_data), np.array(synthetic_labels)

    def train(self, train_folder, my_sensor_folder):
        """Train model on all data sources combined"""
        print("\n" + "=" * 60)
        print("🎯 TRAINING DISTRESS DETECTION MODEL")
        print("=" * 60)

        all_X = []
        all_y = []

        # 1. Load UCI HAR data
        X_uci, y_uci = self.load_uci_har_data(train_folder)
        if len(X_uci) > 0:
            all_X.append(X_uci)
            all_y.append(y_uci)

        # 2. Load your collected sensor data
        X_my, y_my = self.load_my_sensor_data(my_sensor_folder)
        if len(X_my) > 0:
            all_X.append(X_my)
            all_y.append(y_my)

        # 3. Generate synthetic distress data
        X_syn, y_syn = self.generate_synthetic_distress(n_samples=200)
        all_X.append(X_syn)
        all_y.append(y_syn)

        # Combine all data
        if len(all_X) == 0:
            raise ValueError("No data loaded! Check your data folders.")

        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)

        print("\n" + "-" * 60)
        print(f"📊 TOTAL TRAINING DATA: {len(X_combined)} samples")
        print(f"   Features per sample: {X_combined.shape[1]}")

        # Show class distribution
        unique, counts = np.unique(y_combined, return_counts=True)
        print("\n📈 Class Distribution:")
        for cls, count in zip(unique, counts):
            class_name = self.class_names.get(int(cls), f"Unknown-{int(cls)}")
            print(f"   Class {int(cls):2d} ({class_name:20s}): {count:4d} samples")

        # Train model
        print("\n🔄 Training Random Forest classifier...")
        self.model.fit(X_combined, y_combined)

        # Training accuracy
        train_accuracy = self.model.score(X_combined, y_combined)
        print(f"✓ Training accuracy: {train_accuracy * 100:.2f}%")
        print("=" * 60)

        return self.model
        
    def save(self, filepath):
        """Save the trained model to disk"""
        joblib.dump(self.model, filepath)
        print(f"💾 Model saved to {filepath}")
        
    def load(self, filepath):
        """Load a trained model from disk"""
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            print(f"✅ Loaded trained model from {filepath}")
            return True
        return False

    def predict(self, features):
        """Predict class and confidence from features"""
        if len(features) != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} features, got {len(features)}")

        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)[0]
        probabilities = self.model.predict_proba(features_array)[0]
        confidence = np.max(probabilities)

        return int(prediction), confidence

    def get_class_name(self, class_id):
        """Get human readable class name"""
        return self.class_names.get(class_id, f"Unknown-{class_id}")