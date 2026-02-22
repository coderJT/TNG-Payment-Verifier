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

    def extract_text(self, img):
        """
        Extract all text items from the image using EasyOCR.
        Supports numpy array directly to save memory.
        """
        results = self.reader.readtext(img)
        return results

    def perform_ela(self, img, image_path=None, quality=90):
        """
        Perform Error Level Analysis (ELA) on the provided image buffer.
        """
        temp_filename = f"ela_{int(time.time()*1000)}.jpg"
        software_detected = None
        
        # Metadata check (still needs path)
        if image_path:
            try:
                with open(image_path, 'rb') as f:
                    content = f.read()
                    if b'Photoshop' in content: software_detected = "Adobe Photoshop"
                    elif b'GIMP' in content: software_detected = "GIMP"
            except Exception: pass

        if img is None:
            return 0.0, 0.0, software_detected
        
        # Perform ELA in-memory as much as possible
        cv2.imwrite(temp_filename, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        resaved = cv2.imread(temp_filename)
        diff = cv2.absdiff(img, resaved)
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        avg_diff = np.mean(diff)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        hotspot_score = np.std(gray_diff)
            
        return float(avg_diff), float(hotspot_score), software_detected

    def analyze_noise(self, img_bgr, block_size=64):
        """
        Analyze noise floor consistency using isolated pure noise from BGR buffer.
        """
        if img_bgr is None: return 0.0, 0.0
        
        # Convert to gray for noise analysis
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(img, 3)
        pure_noise = cv2.absdiff(img, denoised)
        
        h, w = pure_noise.shape
        variances = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = pure_noise[y:y+block_size, x:x+block_size]
                if np.mean(block) > 0.1:
                    variances.append(np.var(block))
        
        if not variances: return 0.0, 0.0
        avg_noise = np.mean(variances)
        inconsistency = np.std(variances) / (avg_noise + 1e-6)
        return float(avg_noise), float(inconsistency)

    def detect_pca_anomalies(self, img_bgr):
        """
        Detect color anomalies via PCA projection from pre-loaded buffer.
        """
        if img_bgr is None: return 0.0
        
        # PCA needs a lot of memory, so we downsample internally even further
        img = self._downsample_image(img_bgr, max_dim=800)
        rows, cols, colors = img.shape
        flattened = img.reshape((-1, colors)).astype(np.float32)
        
        try:
            cov = np.cov(flattened.T)
            evals, evecs = np.linalg.eigh(cov)
            pc2 = evecs[:, 1]
            projected = np.dot(flattened, pc2)
            projected_map = np.reshape(projected, (rows, cols))
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
        
        for uid in unique_ids:
            cursor.execute("SELECT id FROM transactions WHERE metadata_json LIKE ?", (f'%{uid}%',))
            if cursor.fetchone():
                conn.close()
                return True
                
        conn.close()
        return False

    def parse_data(self, ocr_results):
        """Parse OCR results using patterns and heuristics (Refactor: Concise & Cap-enforced)."""
        lines = [r[1] for r in ocr_results]
        raw, data, used = "\n".join(lines), {}, set()
        
        patterns = {
            "recipient": [r"Transfer To", r"Recipient", r"Pay To", r"Paid To", r"Payee", r"Beneficiary", r"\bTo\b"],
            "amount": [r"Amount", r"Total", r"RM", r"MYR", r"\$"],
            "date_time": [r"Date", r"Time", r"Transaction Date", r"Date & Time"],
            "reference": [r"Ref", r"Reference", r"Wallet Ref", r"Ref No"],
            "transaction_id": [r"Transaction No", r"Transaction ID", r"Txn ID", r"Trace No", r"DuitNow Ref"]
        }
        ui_kb = {"details", "successful", "status", "payment", "receipt", "share", "duitnow", "transaction", "amount", "recipient", "payee", "beneficiary", "done", "pay", "scan", "cimb", "maybank", "rhb", "ewallet", "bank", "tng"}

        def is_bad(val, lbl):
            if any(re.search(r, val, re.I) for p in patterns.values() for r in p) and len(val) < 25:
                return not (lbl == "amount" and re.search(r'RM|MYR|[\$\d\.]{3,}', val))
            return any(b in val.lower() for b in ui_kb)

        # Extraction
        for lbl, regs in patterns.items():
            for i, line in enumerate(lines):
                if any(re.search(r, line, re.I) for r in regs):
                    used.add(i)
                    if lbl == "recipient":
                        # Multi-line name recovery: collect all caps strings in [i-2, i+3]
                        name_parts = []
                        for k in range(max(0, i-2), min(len(lines), i+4)):
                            v = lines[k].strip()
                            if v.isupper() and len(v) >= 3 and not re.search(r'\d', v) and not is_bad(v, lbl):
                                name_parts.append(v)
                                used.add(k)
                        if name_parts:
                            # Join parts, removing duplicates while preserving order
                            seen = set()
                            data[lbl] = " ".join([x for x in name_parts if not (x in seen or seen.add(x))])
                            break
                    else:
                        for j in range(i, min(i+3, len(lines))):
                            v = lines[j].strip()
                            if not v or is_bad(v, lbl): continue
                            if lbl == "amount" and not re.search(r'\d', v): continue
                            data[lbl], _ = v, used.add(j)
                            break
                if lbl in data: break

        # Fallback Recipient
        if "recipient" not in data:
            for i, l in enumerate(lines):
                if any(x in l.lower() for x in ["successful", "rm", "myr", "amount", "$"]):
                    for j in range(i + 1, min(i + 6, len(lines))):
                        v = lines[j].strip()
                        if v.isupper() and len(v) > 5 and not re.search(r'\d', v) and not any(b in v for b in ["SUCCESSFUL", "PAYMENT", "RM", "DETAILS", "SHARE", "RECEIPT", "DONE"]):
                            data["recipient"], _ = v, used.add(j)
                            break
                    if "recipient" in data: break

        # Unknown & Specialized
        ext_v = set(data.values())
        unknown = [l.strip() for i, l in enumerate(lines) if i not in used and len(l.strip()) > 2 and not any(b in l.lower() for b in ui_kb) and l.strip() not in ext_v]
        if unknown: data["unknown_fields"] = unknown

        for r, k in [(r'\d{2}/\d{2}/\d{4}\s+\d{2}[:\.]\d{2}[:\.]\d{2}', "date_time_precise"), (r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', "uuid")]:
            m = re.search(r, raw, re.I if k=="uuid" else 0)
            if m: data[k] = m.group(0) if k=="date_time_precise" else re.findall(r, raw.lower())[0]

        data["layer_1_status"] = "Duplicate" if self.check_duplicate(data) else "Success" if data else "Failure"
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
            
            # Load and downsample ONCE per transaction to save massive memory
            img_bgr_full = cv2.imread(img_path)
            if img_bgr_full is None: continue
            
            img_shared = verifier._downsample_image(img_bgr_full, max_dim=1500)
            del img_bgr_full # Release high-res buffer immediately

            # Layer 1: OCR (using shared buffer)
            ocr_results = verifier.extract_text(img_shared)
            extracted_data = verifier.parse_data(ocr_results)
            extracted_data["image_path"] = img_path
            
            # Layer 2: Image Forensics (using shared buffer)
            ela_score, hotspot_score, s_detected = verifier.perform_ela(img_shared, img_path)
            g_noise, n_inconsis = verifier.analyze_noise(img_shared)
            pca_score = verifier.detect_pca_anomalies(img_shared)
            
            extracted_data.update({
                "ela_score": ela_score,
                "ela_hotspot": hotspot_score,
                "noise_score": g_noise,
                "noise_inconsistency": n_inconsis,
                "pca_score": pca_score,
                "software_detected": s_detected
            })
            
            # Generalized Heuristics (Purely relative internal markers)
            is_suspicious, reasons = False, []

            if s_detected:
                is_suspicious = True
                reasons.append(s_detected)
            
            if n_inconsis > 1.3:
                is_suspicious = True
                reasons.append("Inconsistent Noise")
            
            if hotspot_score > 6.0:
                is_suspicious = True
                reasons.append("ELA Anomaly")

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

