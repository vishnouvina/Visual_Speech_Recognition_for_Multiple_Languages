import os
import torch
from pipelines.model import AVSR
from pipelines.data.data_module import AVSRDataLoader
from pipelines.detectors.mediapipe.detector import LandmarksDetector
import argparse

class InferencePipeline(torch.nn.Module):
    def __init__(self, modality, model_path, model_conf, detector="mediapipe", face_track=False, device="cpu"):
        super(InferencePipeline, self).__init__()
        self.device = device
        self.modality = modality
        self.dataloader = AVSRDataLoader(modality, detector=detector)
        self.model = AVSR(modality, model_path, model_conf, rnnlm=None, rnnlm_conf=None, penalty=0.0, ctc_weight=0.1, lm_weight=0.0, beam_size=40, device=device)
        if face_track and self.modality in ["video", "audiovisual"]:
            self.landmarks_detector = LandmarksDetector()
        else:
            self.landmarks_detector = None

    def process_landmarks(self, data_filename, landmarks_filename):
        if self.modality == "audio":
            return None
        if self.modality in ["video", "audiovisual"]:
            landmarks = self.landmarks_detector(data_filename)
            return landmarks

    def forward(self, data_filename, landmarks_filename=None):
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        transcript = self.model.infer(data)
        return transcript

    def extract_features(self, data_filename, landmarks_filename=None, extract_resnet_feats=False):
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        with torch.no_grad():
            if isinstance(data, tuple):
                enc_feats = self.model.model.encode(data[0].to(self.device), data[1].to(self.device), extract_resnet_feats)
            else:
                enc_feats = self.model.model.encode(data.to(self.device), extract_resnet_feats)
        return enc_feats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AVSR inference and save transcript.")
    parser.add_argument("--modality", type=str, choices=["video", "audiovisual", "audio"], default="video", help="Modality type")
    parser.add_argument("--model_conf", type=str, default="../models/LRS3_V_WER19.1/model.json", help="Path to model config file")
    parser.add_argument("--model_path", type=str, default="../models/LRS3_V_WER19.1/model.pth", help="Path to model weights file")
    parser.add_argument("--video_path", type=str, default="../data/trump.mp4", help="Path to input video/audio file")
    parser.add_argument("--output_path", type=str, default="../output/transcript.txt", help="Path to save transcript")
    parser.add_argument("--face_track", action="store_true", help="Enable face tracking")
    args = parser.parse_args()

    pipeline = InferencePipeline(args.modality, args.model_path, args.model_conf, face_track=args.face_track)
    transcript = pipeline(args.video_path)

    print(f"Transcript: <{transcript}>")

    with open(args.output_path, "w") as f:
        f.write(transcript)
