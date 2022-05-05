# Computational Miniature Mesoscope (CM<sup>2</sup>)
This is an open source repository of the Computational Miniature Mesoscope ([**CM<sup>2</sup>**](https://www.science.org/doi/full/10.1126/sciadv.abb7508)) project in the Computational Imaging Systems Lab ([**CISL**](https://sites.bu.edu/tianlab/)) at Boston University. We aim to develop next-generation “wearable” computational fluorescence microscope that achieves a centimeter-scale field-of-view (FOV) and micron-scale resolution with single-shot 3D imaging capability.
In this repository, we provide: 1) the hardware design of the CM<sup>2</sup> device including 3D printable CAD files of the LED, microlens array housing, and 3D printable freeform illuminators; 2) the Zemax models of the CM<sup>2</sup> system including the ZMX files, spectra data and coating profile; 3) the reconstruction algorithms: ADMM-based 3D deconvolution algorithm with calibrated 3D PSFs and sample measurements, and pre-trained CM2Net for fast, near-isotropic 3D reconstructions.



## Citation
If you find this project useful in your research, please consider citing our paper:

[**Yujia Xue, Ian G. Davison, David A. Boas, and Lei Tian. "Single-shot 3D wide-field fluorescence imaging with a Computational Miniature Mesoscope" Science advances 6, no. 43 (2020): eabb7508.**](https://www.science.org/doi/full/10.1126/sciadv.abb7508)

[**Yujia Xue, Qianwan Yang, Guorong Hu, Kehan Guo, and Lei Tian. "Computational Miniature Mesoscope V2: A deep learning-augmented miniaturized microscope for single-shot 3D high-resolution fluorescence imaging" arXiv: 2205.00123 (2022).**](https://arxiv.org/abs/2205.00123)


## CM<sup>2</sup> V1
Fluorescence microscopes are indispensable to biology and neuroscience. The need for recording in freely behaving animals has further driven the development in miniaturized microscopes (miniscopes). However, conventional microscopes/miniscopes are inherently constrained by their limited space-bandwidth product, shallow depth of field (DOF), and inability to resolve three-dimensional (3D) distributed emitters. Here, we present a Computational Miniature Mesoscope (CM<sup>2</sup>) that overcomes these bottlenecks and enables single-shot 3D imaging across a 7-mm field of view and mm-scale DOF, achieving 7-μm lateral resolution and better than 200-μm axial resolution. The CM<sup>2</sup> features a compact lightweight design that integrates a microlens array for imaging and a light-emitting diode array for excitation. Its expanded imaging capability is enabled by computational imaging that augments the optics by algorithms. We experimentally validate the mesoscopic imaging capability on 3D fluorescent samples. We further quantify the effects of scattering and background fluorescence on phantom experiments.

<p align="center">
  <img src="/Images/Cover.PNG">
</p>

## CM<sup>2</sup> V2
Computational Miniature Mesoscope (CM<sup>2</sup>) is a recently developed computational imaging system that enables single-shot 3D imaging across a wide field-of-view (FOV) using a compact optical platform. In this work, we present CM<sup>2</sup> V2 - an advanced CM<sup>2</sup> system that integrates novel hardware improvements and a new deep learning reconstruction algorithm. The platform features a 3D-printed freeform LED collimator that achieves ~80% excitation efficiency - a ~3x improvement over our V1 design, and a hybrid emission filter design that improves the measurement contrast by >5x. The new computational pipeline includes an accurate and computationally efficient 3D linear shift-variant (LSV) forward model and a novel multi-module CM2Net deep learning model. As compared to the model-based deconvolution in our V1 system, CM2Net achieves ~8x better axial localization and ~1400x faster reconstruction speed. In addition, CM2Net consistently achieves high detection performance and superior axial localization across a wide FOV at a variety of conditions. Trained entirely on our 3D-LSV simulator generated training data set, CM2Net generalizes well to real experiments. We experimentally demonstrate that CM2Net achieves accurate 3D reconstruction of fluorescent emitters across a ~7-mm FOV and 800-μm depth, and provides ~7-μm lateral and ~25-μm axial resolution. We anticipate that this simple and low-cost computational miniature imaging system will be impactful to many large-scale 3D fluorescence imaging applications.

<p align="center">
  <img src="/Images/overview_v2.png">
</p>

## How to use
### 1) Hardware design

The directory 'CAD_models' contains the CAD files of the CM<sup>2</sup>'s V1 and V2 systems, including the LED housing, microlens array housing, LED base plate and the freefrom collimator. All CAD models are 3D printable on lab table-top 3D printers. The subdirectory 'assembly' further provides an assembly of the CM<sup>2</sup> systems shown as below (note that the sensor is not to-scale).
<p align="center">
  <img src="/Images/CAD.PNG"width=600>
</p>
<p align="center">
  <img src="/Images/CAD.PNG"width=600>
</p>

The part list of all optical and electronic components used in the CM<sup>2</sup> prototypes can be found [**here**](https://docs.google.com/spreadsheets/d/1yO0x0pHvZYl-6WYT2bZiUERogTGQaifCt07Zwj_Rsxw/edit?usp=sharing).

### 2) Zemax model

To use the Zemax models, after cloning this repository, copy the coating file "cm2_coating_profiles_ver2.DAT" to the directory "Zemax\Coatings\", copy the spectra files "gfp_emission.spcd" and "led_spectrum_interp.spcd" to the directory "Zemax\Objects\Sources\Spectrum Files\", copy the CAD files (end with '.stl') to the directory "Zemax\Objects\CAD Files\", and then open "CM2_V1_opensource.zos" or "CM2_V2_opensource.zos" in Zemax to view the CM<sup>2</sup> design and ray tracing results. Pre-rendered ray tracing data can be downloaded [**here**](https://drive.google.com/drive/folders/10xuGhUDethVntPEySfjDurRVyuz1iKvr?usp=sharing).

#### CM<sup>2</sup> V1 Zemax model:
<p align="center">
  <img src="/Images/Zemax.PNG"width=800>
</p>

#### CM<sup>2</sup> V2 Zemax model:
<p align="center">
  <img src="/Images/zemax_v2.png"width=800>
</p>

### 3) Model-based 3D deconvolution
The script "cm2_related_code.m" in the "Algorithm" folder provides a demo of CM<sup>2</sup> 3D reconstruction pipeline on a simulated measurement using [down-sampled PSFs](https://drive.google.com/drive/folders/10xuGhUDethVntPEySfjDurRVyuz1iKvr?usp=sharing). A full-scale experimental measurement is also provided under the "Algorithm" direcory, which requires large system memory to run. The GIF file below shows the flying-through of a reconstructed 3D object (a fluorescent fiber sample) from an experimental measurement.

<p align="center">
  <img src="/Images/example_recon.gif"width=600>
</p>

### 4）CM2Net: fast and near-isotropic 3D reconstruction
coming soon...
(https://drive.google.com/drive/folders/10xuGhUDethVntPEySfjDurRVyuz1iKvr?usp=sharing)

## Contact
For further information, please feel free to contact us: Guorong Hu (grhu@bu.edu), Qianwan Yang (yaw@bu.edu), and Prof. Lei Tian (leitian@bu.edu).

## License
This project is licensed under the terms of the MIT license. see the [LICENSE](LICENSE) file for details
