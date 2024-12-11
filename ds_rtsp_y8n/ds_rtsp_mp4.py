import io
import argparse
import sys
import gi
import time

sys.path.append('../')
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst, GstRtspServer
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
import pyds
import platform
import configparser



fps_stream_0 = None

def fps_increment_probe(pad, info, u_data):
    """
    Probe that increments FPS counter for each frame.
    This ensures fps_stream_0 is updated every frame.
    """
    global fps_stream_0
    if fps_stream_0:
        fps_stream_0.get_fps()  # increment frame count internally
    return Gst.PadProbeReturn.OK

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Probe at OSD sink pad to update on-screen text:
    - Remove object IDs from display text.
    - Display current FPS count.
    """
    global fps_stream_0
    fps = fps_stream_0.get_fps()
    if fps is None:
        fps = 0.0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    # Retrieve batch metadata from the GstBuffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

        # Acquire display_meta for FPS text
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]

        # Set FPS display text
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.display_text = "yolov8n_FPS: {:.1f}".format(fps)
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 14
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        # Modify object text to remove IDs, leave only class labels
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            # Set display text to the object label only
            obj_meta.text_params.display_text = obj_meta.obj_label
            l_obj = l_obj.next

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK

def on_source_pad_added(decodebin, pad, data):
    caps = pad.get_current_caps()
    gst_struct = caps.get_structure(0)
    pad_name = gst_struct.get_name()

    print("Received new pad from decodebin: %s" % pad_name)
    if "video" in pad_name:
        nvvidconvsrc = data["nvvidconvsrc"]
        caps_vidconvsrc = data["caps_vidconvsrc"]
        sinkpad = data["streammux_sinkpad"]

        pad.link(nvvidconvsrc.get_static_pad("sink"))
        nvvidconvsrc.link(caps_vidconvsrc)
        caps_vidconvsrc.get_static_pad("src").link(sinkpad)

def main(args):
    global fps_stream_0
    fps_stream_0 = GETFPS(stream_id=0)

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    print("Creating Pipeline")
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    print("Creating uridecodebin")
    source = Gst.ElementFactory.make("uridecodebin", "uri-source")
    if not source:
        sys.stderr.write(" Unable to create uridecodebin \n")
    source.set_property('uri', stream_path)

    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create nvvideoconvert \n")

    caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    if not caps_vidconvsrc:
        sys.stderr.write(" Unable to create capsfilter \n")
    caps_vidconvsrc.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    pgie.set_property('config-file-path', "dstest1_pgie_config.txt")

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create nvtracker \n")

    # Load tracker config
    config = configparser.ConfigParser()
    config.read('dstest2_tracker_config.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width':
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height':
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id':
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file':
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file':
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process':
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame':
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")

    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))

    if codec == "H264":
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        print("Creating H264 Encoder")
    elif codec == "H265":
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        print("Creating H265 Encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property('bitrate', bitrate)
    if is_aarch64():
        encoder.set_property('preset-level', 1)
        encoder.set_property('insert-sps-pps', 1)
        encoder.set_property('bufapi-version', 1)

    if codec == "H264":
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        print("Creating H264 rtppay")
    elif codec == "H265":
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
        print("Creating H265 rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay")

    updsink_port_num = 5400
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not sink:
        sys.stderr.write(" Unable to create udpsink")
    sink.set_property('host', '224.224.255.255')
    sink.set_property('port', updsink_port_num)
    sink.set_property('async', False)
    sink.set_property('sync', 1)

    print("Playing URI: %s " % stream_path)

    print("Adding elements to Pipeline")
    pipeline.add(source)
    pipeline.add(nvvidconvsrc)
    pipeline.add(caps_vidconvsrc)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")

    # Link the elements
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(sink)

    # Add probes
    source.connect("pad-added", on_source_pad_added, {
        "nvvidconvsrc": nvvidconvsrc,
        "caps_vidconvsrc": caps_vidconvsrc,
        "streammux_sinkpad": sinkpad
    })

    # Add FPS increment probe to nvvidconv's source pad (before OSD)
    nvvidconv_src_pad = nvvidconv.get_static_pad("src")
    if nvvidconv_src_pad:
        nvvidconv_src_pad.add_probe(Gst.PadProbeType.BUFFER, fps_increment_probe, 0)
    else:
        sys.stderr.write("Unable to get src pad of nvvidconv for FPS increment\n")

    # Attach the OSD probe for removing IDs and adding FPS text
    nvosd_sink_pad = nvosd.get_static_pad("sink")
    if nvosd_sink_pad:
        nvosd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    else:
        sys.stderr.write("Unable to get sink pad of nvosd\n")

    # Create an event loop and feed GStreamer bus messages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Start RTSP streaming
    rtsp_port_num = 8555
    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )" % (
            updsink_port_num, codec))
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)

    print("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n" % rtsp_port_num)

    # Start playback and listen to events
    print("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)

def parse_args():
    parser = argparse.ArgumentParser(description='RTSP Output Sample Application Help ')
    parser.add_argument("-i", "--input",
                        help="URI to input (RTSP or file)", required=True)
    parser.add_argument("-c", "--codec", default="H264",
                        help="RTSP Streaming Codec H264/H265 , default=H264", choices=['H264', 'H265'])
    parser.add_argument("-b", "--bitrate", default=4000000,
                        help="Set the encoding bitrate ", type=int)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    global codec
    global bitrate
    global stream_path
    codec = args.codec
    bitrate = args.bitrate
    stream_path = args.input
    return 0

if __name__ == '__main__':
    parse_args()
    sys.exit(main(sys.argv))

