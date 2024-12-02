## **ds\_rtsp: YOLO-Based DeepStream Pipeline on Jetson Nano Platform**

This project aims to deploy a YOLO-based DeepStream pipeline optimized for the Jetson Nano platform. The goal is to explore the feasibility and challenges of deploying advanced computer vision models, such as YOLOv8 and YOLOv11, on legacy hardware.

---
### **ds_rtsp_y8n_cam Pipeline**

The `ds_rtsp_y8n_cam` pipeline takes input from a USB camera, run inference using the YOLOv8n model, and provide an RTSP stream as output. 
---

### **Hardware Requirements**

- **Jetson Nano Developer Kit**

  - Preinstalled with [JetPack 4.6.1](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit#intro)
  - Internet access
  - USB webcam

- **Additional Development Environment/Host Machine**

  - SSH client (e.g., OpenSSH, PuTTY)\
    [Remote Development with SSH](https://code.visualstudio.com/docs/remote/ssh)
  - Python-capable environment with dependencies for [Ultralytics](https://github.com/ultralytics/ultralytics?tab=readme-ov-file) preinstalled, including PyTorch.

---

### **Setup**

#### 1. Set Up Remote Development Environment on the Control Machine

The control machine should support file transfers and provide terminal access to the Jetson Nano. [SSH](https://code.visualstudio.com/docs/remote/ssh) is a suitable example for this purpose.

- *Tip: This command will establish a Wi-Fi connection on the Jetson Nano:*
  ```
  sudo nmcli device wifi connect <SSID here> password <password here>
  ```

#### 2. Set Up Remote Development Environment on Control Host

Ensure that:

- SSH is set up to access the Jetson Nano.
- Python dependencies for Ultralytics are installed.

#### 3. Install Dependencies on Jetson Nano

- Complete **DeepStream setup**:

  ```
  sudo apt install libssl1.0.0 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstrtspserver-1.0-0 libjansson4
  ```

- Install **GstRtspServer and Introspection Typelib**:

  ```
  sudo apt update
  sudo apt install python3-gi python3-dev python3-gst-1.0 libgstrtspserver-1.0-0 gstreamer1.0-rtsp \
  libgirepository1.0-dev gobject-introspection gir1.2-gst-rtsp-server-1.0
  ```

- Install **gst-python**:\
  [DeepStream Python Sample Apps](https://docs.nvidia.com/metropolis/deepstream/6.0.1/dev-guide/text/DS_Python_Sample_Apps.html#running-sample-applications)

  ```
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

- Install **deepstream\_python\_apps**:

  ```
  cd <DeepStream 6.0 ROOT>/sources/
  git clone --depth 1 --branch v1.1.1 https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
  ```

- Install **PyDS**:

  ```
  pip3 install pyds
  ```

- Clone **ds\_rtsp** repository to: `<DeepStream 6.0 ROOT>/sources/deepstream_python_apps/apps`

  ```
  git clone https://github.com/alexander11012/ds_rtsp.git
  ```

---


### **Building TensorRT Engine**

We used the [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) repository for the model export process. It provides scripts and libraries that allow us to build a TensorRT engine compatible with DeepStream.

#### **General Approach**

1. **Export** `.pt` **File to** `.onnx` **on Development Machine**
2. **Transfer ONNX File and labels.txt to Jetson Nano**
3. **Convert ONNX to TensorRT Engine using** DeepStream-Yolo
   - Compile libraries using [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) and build a TensorRT engine from the ONNX file.

#### **Detailed Instructions**

##### 1. Copy Conversion Script

- Copy `export_yoloV8.py` from the `DeepStream-Yolo/utils` repository to the `ultralytics` folder.

##### 2. Download YOLOv8 Model

- Download `.pt` file and move it to the `ultralytics` folder.
  ```
  wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
  ```

##### 3. Export Model to ONNX Format

```python
python export_yoloV8.py -w yolov8n.pt --batch 1 --simplify
```

##### 4. Copy Generated Files to Jetson Nano

- Clone the `DeepStream-Yolo` repository on the Jetson Nano and transfer the ONNX and label files there.

##### 5. Compile the Custom Library

```bash
export CUDA_VER=10.2
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

##### 6. Build TensorRT Engine File and Start Pipeline

- Copy `nvdsinfer_custom_impl_Yolo` directory, ONNX model, and label files to the application directory and then navigate there:

  ```
  cd <DeepStream 6.0 ROOT>/sources/deepstream_python_apps/apps/ds_rtsp_y8n_cam
  ```

- Execute the pipeline script to build the TensorRT engine file and start the pipeline.
  *Note: The pipeline may take time to start the first time due to engine building.*

  ```
  python3 ds_rtsp_y8n_cam.py -i /dev/video0
  ```

*Tip: Access the RTSP stream via the Jetson Nano IP at port 8555.*

*Note: ******[Win-RTSP-Player](https://github.com/e1z0/Win-RTSP-Player)****** and VLC player were used to test the RTSP stream output.*


