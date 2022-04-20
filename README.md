# Computational Miniature Mesoscope (CM2)
We provide 1) v1 hardware design: cad files, zmx files, spectra and coating files and 2) v1 algorithm: a toy demo (use downsampled data and PSFs).


### Citation
If you find this project useful in your research, please consider citing our paper:

[**Yujia Xue, Ian G. Davison, David A. Boas, and Lei Tian. "Single-shot 3D wide-field fluorescence imaging with a Computational Miniature Mesoscope." Science advances 6, no. 43 (2020): eabb7508.**](https://www.science.org/doi/full/10.1126/sciadv.abb7508)


### Abstract
Fluorescence microscopes are indispensable to biology and neuroscience. The need for recording in freely behaving animals has further driven the development in miniaturized microscopes (miniscopes). However, conventional microscopes/miniscopes are inherently constrained by their limited space-bandwidth product, shallow depth of field (DOF), and inability to resolve three-dimensional (3D) distributed emitters. Here, we present a Computational Miniature Mesoscope (CM2) that overcomes these bottlenecks and enables single-shot 3D imaging across an 8 mm by 7 mm field of view and 2.5-mm DOF, achieving 7-μm lateral resolution and better than 200-μm axial resolution. The CM2 features a compact lightweight design that integrates a microlens array for imaging and a light-emitting diode array for excitation. Its expanded imaging capability is enabled by computational imaging that augments the optics by algorithms. We experimentally validate the mesoscopic imaging capability on 3D fluorescent samples. We further quantify the effects of scattering and background fluorescence on phantom experiments.

<p align="center">
  <img src="/images/Cover.png">
</p>


### How to use
1) Zemax simulation file

After download the pre-trained weights file, put it under the root directory and run [demo.py](demo.py).

2) Algorithm



## License
This project is licensed under the terms of the MIT license. see the [LICENSE](LICENSE) file for details
