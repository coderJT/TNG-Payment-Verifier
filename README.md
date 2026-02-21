**TNG Payment Verifier**

**Purpose:** To recognize authentic transaction based on TNG Transaction Screenshot provided by payer. It is worth nothing that this method doesn't guarantee 100% authentic transaction as the screenshot may be altered / modified in many ways possible. However, this downside does not render this service unusable, as the ideal screenshot would have sufficient unique transaction metadata (incl. Wallet Ref and Transaction No.), of which may be useful in verifying the transaction authenticity.

---

### **Verification Logic**

#### **Layer 1 - Data Integrity**
This layer ensures the screenshot contains all required transaction details and hasn't been recycled.
*   **OCR Extraction**: Extracts `Transfer To`, `Date/Time`, `Wallet Ref`, `Status`, and `Transaction No`.
*   **Success Condition**: 
    1.  **Completeness**: All 5 required fields must be successfully identified.
    2.  **Uniqueness**: Neither the `Wallet Ref` nor the `Transaction No` can already exist in the database (Duplication Check).

#### **Layer 2 - Image Forensics**
This layer analyzes the mathematical and structural integrity of the image file to detect digital tampering.
*   **Noise Analysis**: Calculates the image's "noise floor" (Laplacian variance).
    *   **Success Condition**: Deviation must be within **15%** of the authentic baseline (~660.0).
*   **Metadata Inspection**: Scans the file binary for digital editing software signatures.
    *   **Success Condition**: No traces of strings like `Adobe Photoshop` or `GIMP` are found.
*   **ELA (Error Level Analysis)**: detects JPEG compression inconsistencies.
    *   **Success Condition**: The average ELA difference score must be **below 2.0**.

---

**How to Run**

1. **Setup Environment**
   Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**
   Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Execution**
   Place transaction screenshots in the `images/` directory and run:
   ```bash
   python3 src/verifier.py
   ```

The script will process images in `images/`, perform OCR and forensic checks, and save results to `transactions.db`.

---

**Potential Improvements:**

1. Integration with cloud databases (e.g., Supabase).
2. Web dashboard for real-time visualization.
3. Machine learning for advanced behavioral analysis.
