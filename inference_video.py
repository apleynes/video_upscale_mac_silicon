import argparse
import io
import subprocess
import ffmpeg
import mimetypes
import glob
import os
import numpy as np
import cv2
import skvideo.io
from tqdm import tqdm
import copy
from PIL import Image
import coremltools as ct
from coremltools.models.neural_network import flexible_shape_utils


# Monkey patch for newer versions of numpy that deprecated np.float and np.int
np.float = float
np.int = int

def get_video_meta_info(video_path):
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    if 'nb_frames' in video_streams[0].keys():
        ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    else: # Use pymediainfo to get the number of frames
        from pymediainfo import MediaInfo
        info = MediaInfo.parse(video_path)
        for track in info.tracks:
            if track.track_type == 'Video':
                ret['nb_frames'] = int(track.frame_count)
                break
    return ret

def get_sub_video(args, num_process, process_idx):
    if num_process == 1:
        return args.input
    meta = get_video_meta_info(args.input)
    duration = int(meta['nb_frames'] / meta['fps'])
    part_time = duration // num_process
    print(f'duration: {duration}, part_time: {part_time}')
    os.makedirs(os.path.join(args.output, f'{args.video_name}_inp_tmp_videos'), exist_ok=True)
    out_path = os.path.join(args.output, f'{args.video_name}_inp_tmp_videos', f'{process_idx:03d}.mp4')
    cmd = [
        args.ffmpeg_bin, f'-i {args.input}', '-ss', f'{part_time * process_idx}',
        f'-to {part_time * (process_idx + 1)}' if process_idx != num_process - 1 else '', '-async 1', out_path, '-y'
    ]
    print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)
    return out_path



class Reader:

    def __init__(self, args, total_workers=1, worker_idx=0):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []  # for image&folder type
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx)
            self.stream_reader = (
                ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='rgb24', vcodec='h264_videotoolbox',
                                                loglevel='error').global_args('-c:v', 'h264_videotoolbox', '-hwaccel', 'videotoolbox').run_async(
                                                    pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
            meta = get_video_meta_info(video_path)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']

        else:
            if self.input_type.startswith('image'):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
                self.paths = paths[num_frame_per_worker * worker_idx:num_frame_per_worker * (worker_idx + 1)]

            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, 'empty folder'
            from PIL import Image
            tmp_img = Image.open(self.paths[0])
            self.width, self.height = tmp_img.size
        self.idx = 0

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        # img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        # img = Image.open(io.BytesIO(img_bytes))
        img = Image.frombytes('RGB', (self.width, self.height), img_bytes)
        return img

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        if self.input_type.startswith('video'):
            return self.get_frame_from_stream()
        else:
            return self.get_frame_from_list()

    def close(self):
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            self.stream_reader.wait()


class Writer:
    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 audio,
                                 video_save_path,
                                 pix_fmt='yuv420p',
                                 vcodec='h264_videotoolbox',
                                 loglevel='error',
                                 acodec='copy').global_args('-c:v', 'h264_videotoolbox', '-hwaccel', 'videotoolbox', '-realtime', '-prio_speed').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 video_save_path, pix_fmt='yuv420p', vcodec='h264_videotoolbox',
                                 loglevel='error').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))

    def write_frame(self, frame):
        if isinstance(frame, np.ndarray):
            frame = frame.astype(np.uint8).tobytes()
        elif isinstance(frame, Image.Image):
            frame = frame.convert('RGB').tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()
        

from torch.utils.data import DataLoader, Dataset

class VideoDataset(Dataset):
    def __init__(self, reader, device):
        self.reader = reader
        self.device = device
        
    def __len__(self):
        return self.reader.nb_frames
    
    def __getitem__(self, idx):
        img = self.reader.get_frame()
        if img is None:
            return None
        return TF.to_tensor(img).to(self.device)
        
def inference_video(args, video_save_path, device=None, total_workers=1, worker_idx=0):
    # session_options = ort.SessionOptions()
    # # session_options.enable_profiling = True
    # session_options.log_verbosity_level = 4
    # session = ort.InferenceSession("upsampler_orig_srgan_2x.onnx", 
    #                                providers=["CoreMLExecutionProvider"], 
    #                                provider_options=[{
    #                                    "ModelFormat": "MLProgram",
    #                                    "MLComputeUnits": "ALL",
    #                                    }],
    #                                sess_options=session_options)
    # upsampler = torch.compile(upsampler)
    
    # model_path = "DASR.mlpackage"
    # model_path = "4xultrasharp256.mlpackage"
    # model_path = "Fast-SRGAN.mlmodel"
    # model_path = "bsrgan.mlmodel"
    # model_path = "realesrganAnime512.mlmodel"
    # model_path = "lesrcnn_x4_128x128.mlmodel"
    # model_path = "aesrgan.mlmodel"
    # model_path = "realcugan/Real-CUGAN/up2x-latest-no-denoise.mlpackage"
    model_path = "realcugan/Real-CUGAN/up2x-latest-denoise1x.mlpackage"
    
    # upsampler = ct.models.MLModel("upsampler_orig_srgan_2x.mlpackage/Data/com.apple.CoreML/model.mlmodel")
    # upsampler = ct.models.MLModel("realesrganAnime512.mlmodel")
    # upsampler = ct.models.MLModel("Fast-SRGAN.mlmodel")
    # upsampler = ct.models.MLModel("lesrcnn_x4_128x128.mlmodel")
    
    if not model_path.endswith(".mlpackage"):
        upsampler = ct.models.MLModel(model_path)
        spec = upsampler.get_spec()
    else:
        mlmodel_path = os.path.join(model_path, "Data", "com.apple.CoreML", "model.mlmodel")
        spec = ct.utils.load_spec(mlmodel_path)

    # get input names
    input_names = [inp.name for inp in spec.description.input]
    input_name = input_names[0]

    # get output names
    output_names = [out.name for out in spec.description.output]
    output_name = output_names[0]
    
    
    reader = Reader(args, total_workers, worker_idx)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    writer = Writer(args, audio, height, width, video_save_path, fps)
    num_frame_batch = 1
    
    if not model_path.endswith(".mlpackage"):
        flexible_shape_utils.add_enumerated_image_sizes(spec, 
                    feature_name=input_name, 
                    sizes=[flexible_shape_utils.NeuralNetworkImageSize(height=height, width=width)])

        weights_dir = os.path.join(model_path, "Data", "com.apple.CoreML", "weights") if model_path.endswith(".mlpackage") else None
        upsampler = ct.models.MLModel(spec, weights_dir=weights_dir)
    else:
        upsampler = ct.models.MLModel(model_path)
    
    # dataset = VideoDataset(reader, device)
    # dataloader = DataLoader(dataset, batch_size=num_frame_batch, shuffle=False, pin_memory=True, num_workers=0)
    
    # example_input = torch.randn(1, 3, height, width).half().to(device)
    # upsampler = torch.jit.trace(upsampler, example_inputs=example_input)
    
    frame_reader = skvideo.io.FFmpegReader(args.input)
    frame_shape = frame_reader.getShape()
    print(frame_shape)
    
    # frame_writer = skvideo.io.FFmpegWriter(video_save_path, outputdict={
    #     'format': 'rawvideo',
    #     'pix_fmt': 'rgb24',
    #     's': f'{args.outscale * width}x{args.outscale * height}',
    #     'framerate': fps,
    #     'vcodec': 'h264_videotoolbox',
    #     'loglevel': 'error',
    #     'acodec': 'copy',
    #     'c:v': 'h264_videotoolbox',
    #     'hwaccel': 'videotoolbox',
    #     'realtime': True,
    #     'prio_speed': True
    # })

    pbar = tqdm(total=frame_shape[0], unit='frame', desc='inference')
    frames = []
    while True:
        # img = reader.get_frame()
        
        img = next(frame_reader.nextFrame())
        if img is None:
                break
        
        try:
            # img = img.transpose(2, 0, 1)[np.newaxis]
            # img = img.astype(np.float32)
            # img = img / 255.0
            # img = Image.fromarray(img).resize((512, 512))
            img = Image.fromarray(img)
            output = upsampler.predict({input_name: img})
            output = output[output_name]
            
            if 'realcugan' in model_path:
                output = output[0].transpose(1, 2, 0).astype(np.uint8)
                output = Image.fromarray(output)
            # output = output['var_1163']
            # output = output[0].squeeze(0)
        except RuntimeError as error:
            print('Error', error)
        else:
            # output = np.flip((output * 255.0).astype(np.uint8).squeeze().transpose(1, 2, 0), axis=-1)
            writer.write_frame(output)
            # frame_writer.writeFrame(np.array(output))
            # frames.append(output)
        pbar.update(1)
        # frames = []
        # for _ in range(num_frame_batch):
        #     img = reader.get_frame()
        #     if img is None:
        #         break
        #     frames.append(TF.to_tensor(img).to(device))
        # if img is None:
        #     break
        # frames = torch.stack(frames, dim=0)
        # with torch.no_grad():
        #     output = upsampler(frames.half())
        # output = (output * 255.0).cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
        # for frame in output:
        #     writer.write_frame(frame)
        # torch.cuda.synchronize(device) if torch.cuda.is_available() else None
        # pbar.update(num_frame_batch)
    # for batch in dataloader:
    #     with torch.no_grad():
    #         output = upsampler(batch.half())
    #     output = (output * 255.0).cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
    #     # for frame in output:
    #     #     writer.write_frame(frame)
    #     pbar.update(num_frame_batch)
    reader.close()
    writer.close()
    
def main():
    """Inference
    It mainly for restoring anime videos.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input video, image or folder')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image', choices=[2, 4])
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored video')
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='The path to ffmpeg')
    parser.add_argument('--num_process_per_gpu', type=int, default=1)
    parser.add_argument('--fps', type=float, default=None, help='FPS of the output video')

    args = parser.parse_args()

    args.input = args.input.rstrip('/').rstrip('\\')
    os.makedirs(args.output, exist_ok=True)

    if mimetypes.guess_type(args.input)[0] is not None and mimetypes.guess_type(args.input)[0].startswith('video'):
        is_video = True
    else:
        is_video = False

    if is_video and args.input.endswith('.flv'):
        mp4_path = args.input.replace('.flv', '.mp4')
        os.system(f'ffmpeg -i {args.input} -codec copy {mp4_path}')
        args.input = mp4_path

    # if args.extract_frame_first and not is_video:
        # args.extract_frame_first = False
        
    args.video_name, file_ext = os.path.splitext(os.path.basename(args.input))
    video_save_path = os.path.join(args.output, f'{args.video_name}_{args.suffix}.{file_ext}')
    inference_video(args, video_save_path)


if __name__ == '__main__':
    main()
