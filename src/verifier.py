import os
import sqlite3
import re
import time
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
        Initialize SQLite database and transactions table.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Initialize transactions table schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transfer_to TEXT,
                date_time TEXT,
                wallet_ref TEXT,
                status TEXT,
                transaction_no TEXT,
                raw_text TEXT,
                layer_1_status TEXT,
                layer_2_status TEXT,
                ela_score REAL,
                noise_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def extract_text(self, image_path):
        """
        Extract all text items from the image using EasyOCR.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        results = self.reader.readtext(image_path)
        return results

    def perform_ela(self, image_path, quality=90):
        """
        Perform Error Level Analysis (ELA) on the image.
        Also checks for digital editing software signatures in metadata.
        Returns (ela_score, software_detected).
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

        # Load original image
        original = cv2.imread(image_path)
        if original is None:
            return 0.0, software_detected

        # Save at a lower quality
        cv2.imwrite(temp_filename, original, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        # Load the lower quality image
        resaved = cv2.imread(temp_filename)
        
        # Calculate absolute difference
        diff = cv2.absdiff(original, resaved)
        
        # Calculate average difference across all channels
        avg_diff = np.mean(diff)
        
        # Clean up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return float(avg_diff), software_detected

    def analyze_noise(self, image_path):
        """
        Basic noise analysis by calculating image variance.
        Inconsistent noise levels often indicate editing.
        """
        img = cv2.imread(image_path, 0) # Grayscale
        if img is None:
            return 0.0
            
        # Calculate Laplacian variance as a proxy for noise level
        noise_level = cv2.Laplacian(img, cv2.CV_64F).var()
        return float(noise_level)

    def check_duplicate(self, wallet_ref, transaction_no):
        """
        Check if the transaction metadata already exists in the database.
        """
        if not wallet_ref and not transaction_no:
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT id FROM transactions WHERE "
        params = []
        conditions = []
        
        if wallet_ref:
            conditions.append("wallet_ref = ?")
            params.append(wallet_ref)
        if transaction_no:
            conditions.append("transaction_no = ?")
            params.append(transaction_no)
            
        if not conditions:
            conn.close()
            return False
            
        cursor.execute(query + " OR ".join(conditions), params)
        result = cursor.fetchone()
        conn.close()
        return result is not None

        results = self.reader.readtext(image_path)
        return results

    def parse_data(self, ocr_results):
        """
        Parse OCR results to extract specific TNG transaction fields.
        """

        data = {
            "transfer_to": None,
            "date_time": None,
            "wallet_ref": "",
            "status": None,
            "transaction_no": None,
            "raw_text": "",
            "layer_1_status": "Failure"
        }

        # Combine items for easier searching
        lines = []
        for (bbox, text, prob) in ocr_results:
            lines.append(text.strip())
        
        raw_content = "\n".join(lines)
        data["raw_text"] = raw_content

        # Simple regex and keyword matching
        for i, line in enumerate(lines):

            # Status
            if "Successful" in line:
                data["status"] = "Successful"
            
            # Transfer To / Payment Details
            if "Transfer To" in line and i + 1 < len(lines):
                data["transfer_to"] = lines[i+1]
            elif "Payment Details" in line and i + 1 < len(lines) and not data["transfer_to"]:
                data["transfer_to"] = lines[i+1]

            # Date/Time (Pattern: DD/MM/YYYY HH.MM.SS or HH:MM:SS)
            date_match = re.search(r'\d{2}/\d{2}/\d{4}\s+\d{2}[:\.]\d{2}[:\.]\d{2}', line)
            if date_match:
                data["date_time"] = date_match.group(0).replace('.', ':')

            # Wallet Ref (Long numeric string)
            if "Wallet Ref" in line:
                ref_parts = []
                for j in range(max(0, i-2), min(len(lines), i+3)):
                    # Only take numbers that aren't also present in the date_time string
                    raw_num = re.sub(r'[^0-9]', '', lines[j])
                    if len(raw_num) > 5:
                        if data["date_time"] and raw_num in data["date_time"].replace(':', '').replace('/', '').replace(' ', ''):
                            continue
                        ref_parts.append(raw_num)
                data["wallet_ref"] = "".join(ref_parts)

            # Transaction No. (UUID-like)
            if "Transaction No." in line or (i > 0 and "Transaction No." in lines[i-1]):
                # Combine surrounding lines but remove the label itself
                context_lines = lines[max(0, i-3):min(len(lines), i+4)]
                filtered_context = [l for l in context_lines if "Transaction No." not in l and "Transaction" not in l and "No." not in l]
                full_context = "".join(filtered_context)
                
                # Clean up OCR errors and normalize
                cleaned_txn = full_context.replace('I', '1').replace('O', '0').replace(' ', '').replace('_', '-').replace('~', '')
                
                # Try to find UUID pattern
                txn_match = re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', cleaned_txn)
                if txn_match:
                    data["transaction_no"] = txn_match.group(0)
                else:
                    # Fallback: look for 32-character hex split by dashes or nothing
                    txn_match = re.search(r'[a-f0-9]{8}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{12}', cleaned_txn)
                    if txn_match:
                        raw_hex = txn_match.group(0).replace('-', '')
                        data["transaction_no"] = f"{raw_hex[:8]}-{raw_hex[8:12]}-{raw_hex[12:16]}-{raw_hex[16:20]}-{raw_hex[20:]}"

        # Determine Layer 1 Status (all required fields must be present and not duplicate)
        required_fields = ["transfer_to", "date_time", "wallet_ref", "status", "transaction_no"]
        is_complete = all(data.get(f) for f in required_fields)
        
        if is_complete:
            if self.check_duplicate(data.get("wallet_ref"), data.get("transaction_no")):
                data["layer_1_status"] = "Duplicate"
            else:
                data["layer_1_status"] = "Success"
        else:
            data["layer_1_status"] = "Failure"
                        
        return data

    def save_to_db(self, data):
        """
        Save extracted data to SQLite database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transactions (
                transfer_to, date_time, wallet_ref, status, transaction_no, 
                raw_text, layer_1_status, layer_2_status, ela_score, noise_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get("transfer_to"),
            data.get("date_time"),
            data.get("wallet_ref"),
            data.get("status"),
            data.get("transaction_no"),
            data.get("raw_text"),
            data.get("layer_1_status"),
            data.get("layer_2_status"),
            data.get("ela_score"),
            data.get("noise_score")
        ))
        conn.commit()
        conn.close()

# Test execution here
if __name__ == "__main__":
    verifier = TNGVerifier()
    test_images = ["payment_sample.PNG", "payment_sample2.PNG", "payment_sample3.PNG", "fraud_sample.jpg", "fraud_sample2.jpg", "fraud_sample3.jpg"]
    
    for img_name in test_images:
        if os.path.exists(img_name):
            print(f"\n--- Processing {img_name} ---")

            # Timing start
            start_time = time.time()
            
            # Layer 1: OCR
            ocr_results = verifier.extract_text(img_name)
            extracted_data = verifier.parse_data(ocr_results)
            
            # Layer 2: Image Forensics
            ela_score, software_detected = verifier.perform_ela(img_name)
            noise_score = verifier.analyze_noise(img_name)
            
            extracted_data["ela_score"] = ela_score
            extracted_data["noise_score"] = noise_score
            extracted_data["software_detected"] = software_detected
            
            # Noise: ~660.0, ELA: ~0.24
            noise_baseline = 660.0
            noise_deviation = abs(noise_score - noise_baseline) / noise_baseline
            
            if noise_deviation > 0.15:
                extracted_data["layer_2_status"] = "Suspicious (Noise Mismatch)"
            elif software_detected:
                 extracted_data["layer_2_status"] = f"Suspicious ({software_detected})"
            elif ela_score > 2.0:
                extracted_data["layer_2_status"] = "Suspicious (ELA Anomaly)"
            else:
                extracted_data["layer_2_status"] = "Success"

            execution_time = time.time() - start_time
            
            print("\nExtracted Transaction Data:")
            for key, value in extracted_data.items():
                if key != "raw_text":
                    print(f"  {key}: {value}")
            
            verifier.save_to_db(extracted_data)
            print(f"\nData saved to transactions.db (Took {execution_time:.2f} seconds Status: L1={extracted_data['layer_1_status']}, L2={extracted_data['layer_2_status']})")
        else:
            print(f"\nImage {img_name} not found.")

