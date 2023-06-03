# Refacer: One-Click Deepfake Multi-Face Swap Tool

Refacer, a simple tool that allows you to create deepfakes with multiple faces with just one click! This project was inspired by [Roop](https://github.com/s0md3v/roop) and is powered by the excellent [Insightface](https://github.com/deepinsight/insightface). Refacer requires no training - just one photo and you're ready to go.

## Demonstration

![demonstration](demo.gif)

[Watch demo video on Youtube](https://youtu.be/mXk1Ox7B244)


[![Watch the video](https://img.youtube.com/vi/mXk1Ox7B244/maxresdefault.jpg)](https://youtu.be/mXk1Ox7B244)


## System Compatibility and Requirements

Refacer has been thoroughly tested on the following operating systems:

| Operating System | CPU Support | GPU Support |
| ---------------- | ----------- | ----------- |
| MacOSX           | ✅         | ❌         |
| Windows          | ✅         | ✅         |
| Linux            | ✅         | ✅         |

The application is compatible with both CPU and GPU (Nvidia CUDA) environments, with the exception of MacOSX which does not currently support GPU (CoreML) usage. 

Please note, we do not recommend using `onnxruntime-silicon` on MacOSX due to an apparent issue with memory management. If you manage to compile `onnxruntime` for Silicon, the program is prepared to use CoreML.


## Usage

With Refacer, you can select a video, choose the face(s) to replace, and the tool does the rest! You'll have a deepfake created in no time.


## Code Release

The code for Refacer will be released in a few hours. Stay tuned for updates.


## Questions?

If you have any questions or issues, feel free to [open an issue](https://github.com/xaviviro/refacer/issues/new) or submit a pull request.


## Recognition Module

The `recognition` folder in this repository is derived from Insightface's GitHub repository. You can find the original source code here: [Insightface Recognition Source Code](https://github.com/deepinsight/insightface/tree/master/web-demos/src_recognition)

This module is used for recognizing and handling face data within the Refacer application, enabling its powerful deepfake capabilities. We are grateful to Insightface for their work and for making their code available.


## Disclaimer

> :warning: This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

> :warning: This software is intended for educational and research purposes only. It is not intended for use in any malicious activities. The author of this software does not condone or support the use of this software for any harmful actions, including but not limited to identity theft, invasion of privacy, or defamation. Any use of this software for such purposes is strictly prohibited.

> :warning: You may only use this software with images for which you have the right to use and the necessary permissions. Any use of images without the proper rights and permissions is strictly prohibited.

> :warning: The author of this software is not responsible for any misuse of the software or for any violation of rights and privacy resulting from such misuse.

> :warning: To prevent misuse, the software contains an integrated protective mechanism that prevents it from working with illegal or similar types of media.

> :warning: By using this software, you agree to abide by all applicable laws, to respect the rights and privacy of others, and to use the software responsibly and ethically.
