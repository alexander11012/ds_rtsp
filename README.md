## **ds\_rtsp: YOLO-Based DeepStream Pipeline on Jetson Nano Platform**

### Objective

Develop a computer vision app, utilizing YOLOv8 to process video feeds from a webcam in real-time, delivering live inference results through an RTSP stream on the Jetson Nano platform.

---

### Configuration plan:

The Jetson Nano operates in headless mode and connects to the local network via Wi-Fi. It processes video feeds from a webcam and streams live inference results to a main workstation through an RTSP feed. Control and monitoring are managed remotely via SSH from the same workstation.

![Scheme](https://github.com/alexander11012/ds_rtsp/blob/main/table_1.png)

---

### Main challenge:
Jetson Nano platform is no more supported by developers, making it prone to compatibility issues when running modern software, not to mention its very limited RAM (2 GB).

---

### Possible Approaches

#### 1. Run YOLOv8 with Ultralytics

Ultralytics is a machine learning framework that provides streamlined tools and pre-trained models, including YOLOv8, for computer vision tasks like object detection and classification.

- **Pros**:
  - User-friendly Docker container with straightforward deployment.
  - Native support for YOLOv8 models.
  - Relatively simple application development.
- **Cons**:
  - High memory consumption due to container overhead.
  - Potential compatibility issues for native execution.
  - Less optimized for live applications.

#### 2. Run via DeepStream SDK

DeepStream SDK is a high-performance framework optimized for deploying AI-powered applications on NVIDIA hardware, particularly for video analytics and streaming.

- **Pros**:
  - Native support for Jetson Nano.
  - High performance in live tasks due to GStreamer libraries.
  - Uses an optimized TensorRT engine for inference.
- **Cons**:
  - Steep learning curve.
  - Lack of native support for YOLOv8 models.

The DeepStream approach was chosen as the only viable option after encountering memory-related crashes with the Ultralytics-Jetson container and numerous compatibility issues. DeepStream addresses the main challenge of limited resources by leveraging TensorRT for efficient inference and GStreamer for lightweight video analytics, making it far more suitable for live applications on low-memory platforms. However, deploying YOLOv8 models with DeepStream is not without its challenges; while YOLOv8 models can be exported to ONNX format, supported by DeepStream, their layer structure and output format are incompatible with it. To overcome this, [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) repository was used for the necessary tools for model export and parsing.

---

### Instructions:

<details>

<summary> Hardware and basic setup Requirements </summary>

- **Jetson Nano Developer Kit**
  - [JetPack 4.6.5](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit#intro) preinstalled, including **DeepStream SDK v6.0.1**
  (Installation via [SDK Manager](https://developer.nvidia.com/sdk-manager) is recommended.)
  - USB webcam
  - USB Wi-Fi adapter

- **Development Machine**
  - Python-capable environment with dependencies for [Ultralytics](https://github.com/ultralytics/ultralytics?tab=readme-ov-file) library preinstalled, including PyTorch.
- **Development Machine** should be configured for remote development and file transfer on **Jetson Nano** in headless mode, via [SSH](https://code.visualstudio.com/docs/remote/ssh) or similar tools.
---
</details>
<details>
<summary>Setup</summary>

#### Dependencies installation on Jetson Nano:
*Note: All path-sensitive commands will assume the DeepStream path to be ```/opt/nvidia/deepstream/deepstream-6.0/``` and will explicitly use this path where applicable.*
- Complete [**DeepStream**](https://docs.nvidia.com/metropolis/deepstream/6.0.1/dev-guide/text/DS_Quickstart.html#install-dependencies) setup and dependencies installation:

  ```bash
  sudo apt update
  sudo apt install libssl1.0.0 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstrtspserver-1.0-0 libjansson4\
  python3-gi python3-dev python3-gst-1.0 libgstrtspserver-1.0-0 gstreamer1.0-rtsp \
  libgirepository1.0-dev gobject-introspection gir1.2-gst-rtsp-server-1.0
  ```

- Install **gst-python**:\
  [DeepStream Python Sample Apps](https://docs.nvidia.com/metropolis/deepstream/6.0.1/dev-guide/text/DS_Python_Sample_Apps.html#running-sample-applications)

  ```bash
  sudo apt-get install python-gi-dev
  export GST_LIBS="-lgstreamer-1.0 -lgobject-2.0 -lglib-2.0"
  export GST_CFLAGS="-pthread -I/usr/include/gstreamer-1.0 -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include"
  git clone https://github.com/GStreamer/gst-python.git
  cd gst-python
  git checkout 1a8f48a
  ./autogen.sh PYTHON=python3
  ./configure PYTHON=python3
  make
  sudo make install
  ```

- Install **deepstream\_python\_apps**
*Note: DeepStream 6.0.1 is supported by version **v1.1.1** of **deepstream\_python\_apps** repository*
    - Clone **deepstream\_python\_apps**
      ```bash
      #path-sensitive!!!
      cd /opt/nvidia/deepstream/deepstream-6.0/sources/
      git clone --depth 1 --branch v1.1.1 https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
      ```
  - Install **DeepStream python bindings**:
      Bindings can be installed via provided [pyds-1.1.1-py3-none-linux_aarch64.whl](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/tag/v1.1.1) file (recommended):
       ```bash
      #run from .whl file directory
      apt install libgirepository1.0-dev libcairo2-dev
      pip3 install pyds-1.1.1-py3-none-linux_aarch64.whl
      ```
    or compiled and installed locally as instructed [here](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/v1.1.1/bindings).

- Install **PyDS**:
    ```bash
    pip3 install pyds
    ```
- Clone **ds\_rtsp** repository to: `/opt/nvidia/deepstream/deepstream-6.0/sources/deepstream_python_apps/apps`
    ```bash
    git clone https://github.com/alexander11012/ds_rtsp.git
    ```
---
</details>
<details>
<summary>Building TensorRT Engine</summary>

We used the [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) repository for the model export process. It provides scripts and libraries that allow us to build a TensorRT engine compatible with DeepStream.

#### **General Approach**

1. **Export** `.pt` **File to** `.onnx` **on Development Machine**
2. **Transfer ONNX File and labels.txt to Jetson Nano**
3. **Convert ONNX to TensorRT Engine using** DeepStream-Yolo
   - Compile libraries using [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) and build a TensorRT engine from the ONNX file.

#### **Detailed Instructions**
- Clone [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) to both **Development Machine** and **Jetson Nano**:

    ```bash
      #Command for Jetson Nano, path is arbitrary
      git clone https://github.com/alexander11012/ds_rtsp.git
    ```
### Note: Following instructions are executed on **Development Machine**:
- Copy `export_yoloV8.py` from the `DeepStream-Yolo/utils` repository to the `ultralytics` folder.

    *Note: Path to `ultralytics` folder depends on your development environment and method of installation. For example, path to ultralytics installed via conda on Windows: "\...\anaconda3\pkgs\ultralytics".*
- Download [YOLOv8 Model](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt) (`.pt` file) and move it to the `ultralytics` folder

- Export Model to ONNX Format

    ```python
    #run this command from ultralytics folder,
    python export_yoloV8.py -w yolov8n.pt --batch 1 --simplify
    ```
- Rename `yolov8n.pt.onnx` file to `yolov8n.onnx`

- Copy `yolov8n.onnx` and label files to **DeepStream-Yolo** folder on **Jetson Nano**

### Note: Following instructions are executed on **Jetson Nano**:

- Compile the **DeepStream-Yolo** Libraries for Yolov8 model integration into DeepStream pipeline:

    ```bash
    #run following command in **DeepStream-Yolo** folder:
    export CUDA_VER=10.2
    make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
    ```

- Copy `nvdsinfer_custom_impl_Yolo` directory, `yolov8n.onnx`, and label files to the application directory: `/opt/nvidia/deepstream/deepstream-6.0/sources/deepstream_python_apps/apps/ds_rtsp_y8n_cam` by default.

- Build TensorRT Engine File by starting the pipeline:
  ```
  #path to app directory for default installation
  cd /opt/nvidia/deepstream/deepstream-6.0/sources/deepstream_python_apps/apps/ds_rtsp_y8n_cam
  python3 ds_rtsp_y8n_cam.py -i /dev/video0
  ```

  *Note: Pipeline may take time to start the first time due to engine building.*

  ```
  python3 ds_rtsp_y8n_cam.py -i /dev/video0
  ```

*Tip: Access the RTSP stream via the Jetson Nano IP at port 8555.*

*Note: ******[Win-RTSP-Player](https://github.com/e1z0/Win-RTSP-Player)****** and VLC player were used to test the RTSP stream output.*
</details>

## Demo


The pipeline and model export method were tested on the Jetson Nano 2GB Developer Kit with jetson_clocks running for maximum performance. The models evaluated included YOLOv8n and YOLOv11n, both with and without an inference frame interval of 2 (where the inference engine skips 2 out of every 3 frames). While frame-skipping significantly improved performance, it also caused bounding box flickering. This issue was mitigated by integrating a tracker into the pipeline.

The videos below showcase recorded RTSP streams, captured on a control machine using VLC player. The video is "sample_720p.mp4" from DeepStream samples.





https://github.com/user-attachments/assets/b8239808-6725-4253-82e3-b14bb0630ead


https://github.com/user-attachments/assets/47139cd1-027e-4889-972a-c2ab91df7323


This is a comparison between the skip and no-skip pipeline configurations running on YOLOv8n and YOLOv11n. The videos are displayed side by side, starting from the same point in the sequence. It is evident that the performance difference between these configurations is substantial and far exceeds the difference observed between the two models when running the frame-skip configuration.



https://github.com/user-attachments/assets/47c8d650-227d-4335-a679-a61051fa184b

