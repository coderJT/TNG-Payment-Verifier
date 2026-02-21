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
            "raw_text": ""
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
                        
        return data

    def save_to_db(self, data):
        """
        Save extracted data to SQLite database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transactions (transfer_to, date_time, wallet_ref, status, transaction_no, raw_text)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            data.get("transfer_to"),
            data.get("date_time"),
            data.get("wallet_ref"),
            data.get("status"),
            data.get("transaction_no"),
            data.get("raw_text")
        ))
        conn.commit()
        conn.close()

# Test execution here
if __name__ == "__main__":
    verifier = TNGVerifier()
    sample_image = "payment_sample.jpeg"
    
    if os.path.exists(sample_image):
        print(f"Processing {sample_image}")

        start_time = time.time()
        ocr_results = verifier.extract_text(sample_image)
        extracted_data = verifier.parse_data(ocr_results)
        execution_time = time.time() - start_time
        
        print("\nExtracted Transaction Data:")
        for key, value in extracted_data.items():
            if key != "raw_text":
                print(f"  {key}: {value}")
        
        verifier.save_to_db(extracted_data)
        print(f"\nData saved to transactions.db (Took {execution_time:.2f} seconds)")
    else:
        print(f"Sample image {sample_image} not found in current directory.")
