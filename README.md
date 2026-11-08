\# VoltVision AI ⚡



\### AI-Powered Power Line Inspection and Defect Detection System



VoltVision AI is a deep learning-based application designed to automate the visual inspection of electrical power-line infrastructure using computer vision.



\## Key Features



\- Automated power-line component detection.

\- Faster R-CNN with ResNet-50 FPN architecture.

\- Image and video-based inspection.

\- Component-level inspection and analysis.

\- Prediction and ground-truth comparison.

\- Detailed evaluation reports.

\- Inspection history and report generation.

\- Interactive Streamlit dashboard.



\## Technology Stack



\- Python

\- PyTorch

\- Torchvision

\- Faster R-CNN

\- ResNet-50 FPN

\- Streamlit

\- OpenCV

\- Pandas

\- NumPy



\## Project Structure



```text

VoltVision-AI/

├── app.py

├── config.py

├── modules/

├── data/

├── dataset\_inspection/

├── ground\_truth\_inspection/

├── prediction\_comparison/

├── detailed\_evaluation/

├── notebooks/

├── requirements.txt

└── README.md

```



\## Installation



1\. Clone the repository:



```bash

git clone https://github.com/YOUR\_USERNAME/VoltVision-AI.git

cd VoltVision-AI

```



2\. Create a virtual environment:



```bash

python -m venv venv

```



3\. Activate it:



```bash

venv\\Scripts\\activate

```



4\. Install dependencies:



```bash

pip install -r requirements.txt

```



5\. Place the trained model checkpoint in the configured model directory. The application expects `checkpoint\_epoch\_10.pth` by default, or a path specified using `POWERLINE\_MODEL\_PATH`.



6\. Start the application:



```bash

streamlit run app.py

```



\## Model Architecture



\*\*Faster R-CNN + ResNet-50 FPN\*\*



The system uses a two-stage object detection architecture for identifying power-line components.



\## Evaluation



The project includes class-wise evaluation metrics, detailed prediction matching, and ground-truth comparison results.



\## Project Status



Developed as a deep learning and computer vision project.



\## Author



Developed by Priyadarshani Dash.



