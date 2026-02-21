**TNG Payment Verifier**

**Purpose:** To recognize authentic transaction based on TNG Transaction Screenshot provided by payer. It is worth nothing that this method doesn't guarantee 100% authentic transaction as the screenshot may be altered / modified in many ways possible. However, this downside does not render this service unusable, as the ideal screenshot would have sufficient unique transaction metadata (incl. Wallet Ref and Transaction No.), of which may be useful in verifying the transaction authenticity.


**Detection Workflow:**

1. Layer 1 - OCR Text Extraction (Transfer To, Date/Time, Wallet Ref, Status, Transaction No.) -> Store in DB (e.g. Supabase)
2. Layer 2 - OpenCV Template Matching
3. Layer 3 - Image Forensics (ELA, Noise Analysis)


**What to do if flagged as fraudulent?**

1. Notify admin immediately.


**Production Workflow:**

1. Host app accepts user's payment input (through screenshot).
2. Detection flow executes. (< 5 seconds ideally)
3. Admin will receive status updates in real-time.


**Potential Bottlenecks:**

1. How long will the detection workflow execute?
2. How accurate is the entire workflow?


**Potential Improvements:**

1. Usage of machine learning model (however requires big bulk of data.)
