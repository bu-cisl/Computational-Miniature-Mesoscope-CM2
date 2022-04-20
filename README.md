# Computational Miniature Mesoscope (CM2)
We provide 1) v1 hardware design: cad files, zmx files, spectra and coating files and 2) v1 algorithm: a toy demo (use downsampled data and PSFs).


### Citation
If you find this project useful in your research, please consider citing our paper:

[**Yujia Xue, Ian G. Davison, David A. Boas, and Lei Tian. "Single-shot 3D wide-field fluorescence imaging with a Computational Miniature Mesoscope." Science advances 6, no. 43 (2020): eabb7508.**](https://www.science.org/doi/full/10.1126/sciadv.abb7508)


### Abstract
Fluorescence microscopes are indispensable to biology and neuroscience. The need for recording in freely behaving animals has further driven the development in miniaturized microscopes (miniscopes). However, conventional microscopes/miniscopes are inherently constrained by their limited space-bandwidth product, shallow depth of field (DOF), and inability to resolve three-dimensional (3D) distributed emitters. Here, we present a Computational Miniature Mesoscope (CM2) that overcomes these bottlenecks and enables single-shot 3D imaging across an 8 mm by 7 mm field of view and 2.5-mm DOF, achieving 7-μm lateral resolution and better than 200-μm axial resolution. The CM2 features a compact lightweight design that integrates a microlens array for imaging and a light-emitting diode array for excitation. Its expanded imaging capability is enabled by computational imaging that augments the optics by algorithms. We experimentally validate the mesoscopic imaging capability on 3D fluorescent samples. We further quantify the effects of scattering and background fluorescence on phantom experiments.

<p align="center">
  <img src="/Images/Cover.PNG">
</p>


### How to use
1) Zemax simulation file

   After downloading the "Zemax_models" folder, put "CM2_V1_opensource.CFG" under the directory "Zemax\Configs", put "cm2_coating_profiles_ver2.DAT" under the      directory "Zemax\Coatings", put "gfp_emission.spcd" and "led_spectrum_interp.spcd" under the directory "Zemax\Objects\Sources\Spectrum Files", put "led_housing.stl",  "mla_housing.stl", and "zemax_mla_aperture.stl" under the directory "Zemax\Objects\CAD Files", and run "CM2_V1_opensource.zos".

<img src="/Images/CAD.PNG"><img src="/Images/Zemax.PNG">

2) Algorithm



## License
This project is licensed under the terms of the MIT license. see the [LICENSE](LICENSE) file for details
