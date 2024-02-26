<header>

<!--
  <<< Author notes: Course header >>>
  Include a 1280×640 image, course title in sentence case, and a concise description in emphasis.
  In your repository settings: enable template repository, add your 1280×640 social image, auto delete head branches.
  Add your open source license, GitHub uses MIT license.
-->


<img src="https://raw.githubusercontent.com/rkassila/Medical_AImaging/master/aimaging/interface/images/title_image.png" alt="Medical AImaging" width="1280" align="right">

# Medical AImaging

_A deep learning solution designed for the detection of diseases in Magnetic Resonance (MR) and Computed Tomography (CT) images, utilizing Convolutional Neural Networks (CNN)._

## Project Overview

Trained on the [Rad Image Net dataset](https://www.radimagenet.com), this project is an exploration into medical diagnostics using deep learning. 
Medical AImaging was our Le Wagon bootcamp project focused on detecting 36 diseases across 5 organs. 

### Demo

[Click here](https://youtu.be/I43Ln32OAMs?t=1076&si=Rjq8IJsQYe_u5sY1) to watch the demo.

---
### Built With

- **Tensorflow:** Leveraged as a foundational framework for developing and implementing advanced CNNs.

- **FastAPI:** Independently incorporated to optimize functionality and enhance overall application performance.

- **Streamlit:** Chosen for deployment, providing a user-friendly interface and facilitating a smooth deployment process.

---
## Images

<table>
  <tr>
    <td align="top"><img src="https://raw.githubusercontent.com/rkassila/Medical_AImaging/master/aimaging/interface/images/shap_image.png" alt="Shap" width="500"></td>
    <td align="top"><img src="https://raw.githubusercontent.com/rkassila/Medical_AImaging/master/aimaging/interface/images/ai_vision.png" alt="AI Vision" width="500"></td>
  </tr>
  <tr>
    <td align="center">Organ Detection</td>
    <td align="center">Disease Detection</td>
  </tr>
</table>

---
## How to use ?
<p>
  <i>To be able to test the app, I've created small models : original models being > 2 Gb in total (over GitHub limit). <br/>
    Those small models accuracy (46-70%) is not as good as the original models (see below for explanations) but is offering sufficient performance considenring the reduction of size : original diseases classifiers were over 700Mb each + 200Mb for binary detection (diseased or not), small models are 2.5Mb for both usages.<br/> Images displayed on this ReadMe have been made with original models.</i>

- Download the repository
```bash
git clone https://github.com/rkassila/Medical_AImaging.git
```
```bash
pip install -r requirements.txt
pip install .
sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 -y
run-api
streamlit run aimaging/interface/main.py
```

</p>

<br/>

<p><b>Original models : </b> Organ Classifier (99.9% accuracy) > Disease detection (90~99% accuracy) > Disease classifier (55~90% accuracy)</p>
<p><b>Small models : </b> Organ Classifier (99.9% accuracy) > Disease classifier with healthy included (46~70% accuracy)</p>

<footer>
</footer>
