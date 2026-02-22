import os
import sqlite3
import re
import time
import json
import easyocr
import cv2
import numpy as np

class TNGVerifier:
    """
    Verifier class for TNG transactions.
    """
    def __init__(self, db_path=None):
        if db_path is None:
            # Database configuration (default uses SQLite)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.db_path = os.path.join(base_dir, "transactions.db")
        else:
            self.db_path = db_path
            
        self.reader = easyocr.Reader(['en'])
        self._init_db()

    def _init_db(self):
        """
        Initialize SQLite database with a dynamic JSON-based schema.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                metadata_json TEXT, -- All extracted fields stored as JSON
                layer_1_status TEXT,
                layer_2_status TEXT,
                ela_score REAL,
                ela_hotspot REAL,
                noise_score REAL,
                noise_inconsistency REAL,
                pca_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def _downsample_image(self, img, max_dim=1500):
        """
        Downsample image for forensic analysis to reduce memory footprint.
        """
        if img is None:
            return None
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return img

    def extract_text(self, image_path):
        """
        Extract all text items from the image using EasyOCR.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Reader is initialized in __init__, just use it
        results = self.reader.readtext(image_path)
        return results

    def perform_ela(self, image_path, quality=90):
        """
        Perform Error Level Analysis (ELA) on the image.
        """
        temp_filename = "ela_temp.jpg"
        software_detected = None
        
        # Metadata check
        try:
            with open(image_path, 'rb') as f:
                content = f.read()
                if b'Photoshop' in content:
                    software_detected = "Adobe Photoshop"
                elif b'GIMP' in content:
                    software_detected = "GIMP"
        except Exception:
            pass

        # Load and downsample for memory efficiency
        original_full = cv2.imread(image_path)
        if original_full is None:
            return 0.0, 0.0, software_detected
        
        original = self._downsample_image(original_full)
        del original_full # Free memory

        # Save at a lower quality and calculate difference
        cv2.imwrite(temp_filename, original, [cv2.IMWRITE_JPEG_QUALITY, quality])
        resaved = cv2.imread(temp_filename)
        diff = cv2.absdiff(original, resaved)
        
        # Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        # Calculate average difference (global ELA)
        avg_diff = np.mean(diff)
        
        # Calculate localized anomalies (standard deviation of the ELA map)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        hotspot_score = np.std(gray_diff)
            
        return float(avg_diff), float(hotspot_score), software_detected

    def analyze_noise(self, image_path, block_size=64):
        """
        Analyze noise floor consistency using isolated pure noise.
        """
        # Load and downsample to save memory
        img_full = cv2.imread(image_path, 0)
        if img_full is None:
            return 0.0, 0.0
        
        img = self._downsample_image(img_full)
        del img_full

        denoised = cv2.medianBlur(img, 3)
        pure_noise = cv2.absdiff(img, denoised)
        
        h, w = pure_noise.shape
        variances = []
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = pure_noise[y:y+block_size, x:x+block_size]
                if np.mean(block) > 0.1:
                    variances.append(np.var(block))
        
        if not variances:
            return 0.0, 0.0

        avg_noise = np.mean(variances)
        inconsistency = np.std(variances) / (avg_noise + 1e-6)
        
        return float(avg_noise), float(inconsistency)

    def detect_pca_anomalies(self, image_path):
        """
        Use Principal Component Analysis to detect localized color-space manipulation.
        """
        img_full = cv2.imread(image_path)
        if img_full is None:
            return 0.0
            
        # Heavy downsampling for PCA to keep memory usage low (it creates large cov matrices)
        img = self._downsample_image(img_full, max_dim=800)
        del img_full
        
        rows, cols, colors = img.shape
        flattened = img.reshape((-1, colors)).astype(np.float32)
        
        # PCA projection logic (2nd principal component)
        try:
            cov = np.cov(flattened.T)
            evals, evecs = np.linalg.eigh(cov)
            # eigh returns sorted, so index 1 (middle) is the 2nd component
            pc2 = evecs[:, 1]
            projected = np.dot(flattened, pc2)
            projected_map = np.reshape(projected, (rows, cols))
            
            # Anomaly score is the normalized standard deviation of the projected map
            # Authentic images should be relatively uniform in this projection
            local_variance = np.std(projected_map) / (np.abs(np.mean(projected_map)) + 1e-6)
            return float(local_variance)
        except Exception:
            return 0.0

    def check_duplicate(self, extracted_data):
        """
        Check if any unique IDs in the extracted data already exist in the database.
        """
        # Look for things that look like unique IDs (UUIDs, long numeric strings)
        unique_ids = []
        for val in extracted_data.values():
            if not isinstance(val, str): continue
            
            # UUIDs
            if re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', val.lower()):
                unique_ids.append(val)
            # Long numeric refs (> 10 digits)
            elif re.search(r'\d{12,}', val):
                unique_ids.append(val)

        if not unique_ids:
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # We search if these IDs appear anywhere in the metadata_json of existing records
        for uid in unique_ids:
            cursor.execute("SELECT id FROM transactions WHERE metadata_json LIKE ?", (f'%{uid}%',))
            if cursor.fetchone():
                conn.close()
                return True
                
        conn.close()
        return False

    def parse_data(self, ocr_results):
        """
        Parse OCR results using generalized patterns to support various payment methods.
        """
        data = {}
        lines = [r[1] for r in ocr_results]
        raw_content = "\n".join(lines)
        
        # Generalized Patterns
        patterns = {
            "recipient": [r"Transfer To", r"Recipient", r"Pay To", r"Paid To", r"Payee", r"Beneficiary"],
            "amount": [r"Amount", r"Total", r"RM", r"\$"],
            "date_time": [r"Date", r"Time", r"Transaction Date", r"Date & Time"],
            "reference": [r"Ref", r"Reference", r"Wallet Ref", r"Ref No"],
            "transaction_id": [r"Transaction No", r"Transaction ID", r"Txn ID", r"Trace No", r"DuitNow Ref"]
        }

        # Dynamic Extraction with index tracking
        used_indices = set()
        for label, regexes in patterns.items():
            for i, line in enumerate(lines):
                if any(re.search(r, line, re.IGNORECASE) for r in regexes):
                    used_indices.add(i)
                    # Try to find the value in the CURRENT or next few items
                    for j in range(i, min(i+3, len(lines))):
                        val = lines[j].strip()
                        if not val: continue
                        
                        # SKIP logic
                        is_ui_label = False
                        for all_regexes in patterns.values():
                            if any(re.search(r, val, re.IGNORECASE) for r in all_regexes):
                                if len(val) < 25: 
                                    is_ui_label = True
                                    break
                        
                        ui_blacklist = ["details", "successful", "status", "payment", "transfer to wallet", 
                                        "receipt", "share", "duitnow", "transaction", "amount", "recipient", "payee", "beneficiary"]
                        if is_ui_label or any(b in val.lower() for b in ui_blacklist):
                            if label == "amount" and re.search(r'RM|[\$\d\.]{3,}', val):
                                pass 
                            else:
                                continue
                            
                        # VALUE VALIDATION
                        if label == "amount" and not re.search(r'\d', val): continue
                        if label == "recipient":
                            if any(x in val.lower() for x in ["wallet", "ref", "time", "date"]): continue
                            if len(val) < 3 or not re.search(r'[A-Za-z]', val): continue
                        
                        if val:
                            data[label] = val
                            used_indices.add(j)
                            break
                    if label in data:
                        break

        # FALLBACK: Recipient recovery
        if "recipient" not in data:
            for i, line in enumerate(lines):
                if any(x in line.lower() for x in ["successful", "rm", "amount", "$"]):
                    for j in range(i + 1, min(i + 4, len(lines))):
                        candidate = lines[j].strip()
                        if (len(candidate) > 5 and candidate.isupper() and 
                            re.search(r'[A-Z]', candidate) and 
                            not re.search(r'\d', candidate)):
                            
                            ui_blacklist = ["SUCCESSFUL", "PAYMENT", "RM", "DETAILS", "SHARE", "RECEIPT", "DONE"]
                            if not any(b in candidate for b in ui_blacklist):
                                data["recipient"] = candidate
                                used_indices.add(j)
                                break
                    if "recipient" in data:
                        break

        # COLLECT UNKNOWN FIELDS
        unknown = []
        ui_blacklist = ["details", "successful", "status", "payment", "receipt", "share", "done", "pay", "scan", "receipt"]
        extracted_values = set(data.values())
        for i, line in enumerate(lines):
            val = line.strip()
            if i not in used_indices and len(val) > 2:
                # Filter out pure digits (usually refs/dates caught elsewhere) or known UI words
                # and skip values already assigned to a primary field
                if not any(b in val.lower() for b in ui_blacklist) and val not in extracted_values:
                    unknown.append(val)
        if unknown:
            data["unknown_fields"] = unknown

        # Specific heavy-lifting for complex fields
        # Date/Time
        date_match = re.search(r'\d{2}/\d{2}/\d{4}\s+\d{2}[:\.]\d{2}[:\.]\d{2}', raw_content)
        if date_match: data["date_time_precise"] = date_match.group(0)

        # UUID / Transaction IDs
        uuids = re.findall(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', raw_content.lower())
        if uuids: data["uuid"] = uuids[0]

        # Logic for Layer 1 Status
        if len(data) >= 1: # Heuristic: if we found at least 3 key fields, consider it a parse success
            if self.check_duplicate(data):
                data["layer_1_status"] = "Duplicate"
            else:
                data["layer_1_status"] = "Success"
        else:
            data["layer_1_status"] = "Failure"
            
        return data

    def save_to_db(self, data):
        """
        Save dynamic metadata and forensic results to SQLite.
        """
        # Separate metadata from status/forensics
        metadata = {k: v for k, v in data.items() if k not in [
            "image_path", "layer_1_status", "layer_2_status", 
            "ela_score", "ela_hotspot", "noise_score", "noise_inconsistency", "pca_score"
        ]}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transactions (
                image_path, metadata_json, layer_1_status, layer_2_status, 
                ela_score, ela_hotspot, noise_score, noise_inconsistency, pca_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get("image_path"),
            json.dumps(metadata),
            data.get("layer_1_status"),
            data.get("layer_2_status"),
            data.get("ela_score"),
            data.get("ela_hotspot"),
            data.get("noise_score"),
            data.get("noise_inconsistency"),
            data.get("pca_score")
        ))
        conn.commit()
        conn.close()

# Test execution here
if __name__ == "__main__":
    verifier = TNGVerifier()
    test_dir = "images"
    
    if os.path.exists(test_dir):
        image_extensions = ('.jpg', '.jpeg', '.png', '.PNG')
        test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(image_extensions)]
        
        print(f"Found {len(test_images)} images in {test_dir}/")

        for img_path in sorted(test_images):
            print(f"\n--- Processing {img_path} ---")

            # Timing start
            start_time = time.time()
            
            # Layer 1: OCR
            ocr_results = verifier.extract_text(img_path)
            extracted_data = verifier.parse_data(ocr_results)
            extracted_data["image_path"] = img_path
            
            # Layer 2: Image Forensics
            ela_score, hotspot_score, software_detected = verifier.perform_ela(img_path)
            global_noise, noise_inconsistency = verifier.analyze_noise(img_path)
            pca_score = verifier.detect_pca_anomalies(img_path)
            
            extracted_data["ela_score"] = ela_score
            extracted_data["ela_hotspot"] = hotspot_score
            extracted_data["noise_score"] = global_noise
            extracted_data["noise_inconsistency"] = noise_inconsistency
            extracted_data["pca_score"] = pca_score
            extracted_data["software_detected"] = software_detected
            
            # Generalized Heuristics (Purely relative internal markers)
            is_suspicious = False
            reasons = []

            if software_detected:
                is_suspicious = True
                reasons.append(software_detected)
            
            # Localized noise shifts (Normal images usually have < 1.0 inconsistency)
            if noise_inconsistency > 1.3:
                is_suspicious = True
                reasons.append("Inconsistent Noise")
            
            # Localized compression hotspots (High variance in ELA map)
            if hotspot_score > 6.0:
                is_suspicious = True
                reasons.append("ELA Anomaly")

            # PCA Color Anomaly
            if pca_score > 0.25:
                is_suspicious = True
                reasons.append("Color Tampering")
            
            if is_suspicious:
                 extracted_data["layer_2_status"] = f"Suspicious ({', '.join(reasons)})"
            else:
                extracted_data["layer_2_status"] = "Success"

            execution_time = time.time() - start_time
            
            print("\nExtracted Transaction Data (Dynamic):")
            for key, value in extracted_data.items():
                if key != "metadata_json":
                    print(f"  {key}: {value}")
            
            verifier.save_to_db(extracted_data)
            print(f"\nData saved to transactions.db (Took {execution_time:.2f} seconds Status: L1={extracted_data['layer_1_status']}, L2={extracted_data['layer_2_status']})")
    else:
        print(f"Test directory {test_dir} not found.")

