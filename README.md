# stegoseek
A command-line tool for detecting hidden data in images using classical steganalysis techniques. 

This project combines multiple statistical, spatial, and frequency-domain tests to identify potential steganographic content in both raw and JPEG images. It can be used to check whether an image or a batch of images have ossible hidden data or not. This is a basic program it needs to be more refined. But it can be helpful to select images worth checking.


📦 Installation
Clone the repository:
git clone https://github.com/yourusername/stegdetect.git
cd stegdetect


▶️ Usage
🔹 Analyze a single image
python steg_detect.py image.png
🔹 Analyze a directory
python steg_detect.py images/
🔹 Limit number of files
python steg_detect.py images/ --limit 50
🔹 Random sampling 
python steg_detect.py images/ --limit 50 --random


🧠 Verdict Logic
CLEAN → No suspicious patterns detected
SUSPICIOUS -> Investigation worthy
HIGH POSSIBILITY -> Very likely to have steganography
⚠️ This tool is designed for investigation and triage, not definitive proof.


🚀 Features
Detects spatial-domain steganography and some JPEG-domain anomalies
Works based on statistical tests
Supports single image and batch directory analysis
Provides human-readable reasons for detection


🧪 Detection Methods
Method	Description
RS Analysis	Detects statistical imbalance in pixel groups
Sample Pair Analysis (SPA)	Estimates embedding rate from pixel pairs
DCT LSB Test	Detects anomalies in JPEG frequency coefficients
Double Compression Test	Identifies recompression artifacts
Histogram Analysis	Detects even-odd distribution anomalies
LSB Plane Analysis	Examines randomness of least significant bits
Noise Residual Analysis	Detects structured noise in smooth regions
Channel Entropy	Finds abnormal entropy differences across channels
EOF Data Check	Detects hidden data appended after file end


⚠️ Limitations
Not effective against modern adaptive steganography
May produce false positives on:
highly compressed images
noisy or textured images
Some tests detect artifacts, not stego directly


📌 Future Improvements
Detection of Transform Domain Steganography
ML model to detect steganography


⭐ Contributiion/Collaboration
suggestions, collaboration, projects are welcome.
