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

#### **Layer 2 - Image Forensics (Generalized)**
This layer analyzes the mathematical and structural integrity of the image file to detect digital tampering without relying on device-specific baselines.
*   **PCA Color Analysis**: Projects colors onto the 2nd principal component to detect localized color-space manipulation.
    *   **Success Condition**: PCA standard deviation score must be **below 0.25**.
*   **Metadata Inspection**: Scans the file binary for digital editing software signatures (Photoshop/GIMP).
    *   **Success Condition**: No software metadata is detected.
*   **Noise Inconsistency**: Measures the standard deviation of 'pure noise' variance across different blocks of the image.
    *   **Success Condition**: Inconsistency score must be **below 1.2**.
*   **ELA (Error Level Analysis)**: detects JPEG compression hotspots.
    *   **Success Condition**: ELA hotspot score must be **below 5.0**.

---

### **Memory Optimization**
The verifier is optimized to handle large batches of high-resolution images:
- **Dynamic Downsampling**: Forensic analysis is performed on a downsampled version of the image to keep RAM usage low.
- **Resource Management**: explicit memory cleanup after processing each image.

---

**How to Run**

1. **Setup Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execution**
   Place transaction screenshots in the `images/` directory and run:
   ```bash
   python3 src/verifier.py
   ```

The script will process images in `images/`, perform OCR and forensic checks, and save results to `transactions.db`.
