**TNG Payment Verifier**

**Purpose:** To recognize authentic transaction based on TNG Transaction Screenshot provided by payer. It is worth nothing that this method doesn't guarantee 100% authentic transaction as the screenshot may be altered / modified in many ways possible. However, this downside does not render this service unusable, as the ideal screenshot would have sufficient unique transaction metadata (incl. Wallet Ref and Transaction No.), of which may be useful in verifying the transaction authenticity.

**Detection Workflow:**

1. Layer 1 - OCR Text Extraction (Transfer To, Date/Time, Wallet Ref, Status, Transaction No.) -> Store in DB (SQLite) -> Duplication Check
2. Layer 2 - Image Forensics (ELA, Noise Analysis)

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

**Potential Improvements:**

1. Integration with cloud databases (e.g., Supabase).
2. Web dashboard for real-time visualization.
3. Machine learning for advanced behavioral analysis.
