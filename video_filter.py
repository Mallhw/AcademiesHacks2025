import cv2
import numpy as np
import pyvirtualcam
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.BatchNorm2d(in_features),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.BatchNorm2d(in_features)
        )
    
    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, ngf=32, n_residual_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, 7),
            nn.BatchNorm2d(ngf),
            nn.ReLU()
        ]
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU()
            ]
            in_features = out_features
            out_features = in_features * 2
        for _ in range(n_residual_blocks):
            model.append(ResidualBlock(in_features))
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU()
            ]
            in_features = out_features
            out_features = in_features // 2
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

preprocess = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0,1)
    return transforms.ToPILImage()(tensor)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_pretrained_generator(model_path='state_dict.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(ngf=32, n_residual_blocks=9).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator, device

generator, device = load_pretrained_generator('state_dict.pth')

def add_beard(face, alpha):
    L, k, x0 = 1.0, 5.0, 0.5
    logistic_factor = L / (1 + np.exp(-k * (alpha - x0)))
    max_ratio = 0.5
    beard_height = int(logistic_factor * max_ratio * face.shape[0])
    if beard_height <= 0:
        return face
    beard_img = cv2.imread("beard.png", cv2.IMREAD_UNCHANGED)
    if beard_img is None:
        print("Beard image not found.")
        return face
    face_width = face.shape[1]
    beard_resized = cv2.resize(beard_img, (face_width, beard_height), interpolation=cv2.INTER_AREA)
    if beard_resized.shape[2] == 4:
        beard_rgb = beard_resized[:, :, :3]
        beard_alpha = beard_resized[:, :, 3].astype(float) / 255.0
        beard_alpha = beard_alpha[..., np.newaxis]
    else:
        beard_rgb = beard_resized
        beard_alpha = np.ones((beard_resized.shape[0], beard_resized.shape[1], 1), dtype=float)
    y_offset = face.shape[0] - beard_height
    roi = face[y_offset:face.shape[0], 0:face_width]
    blended = (beard_alpha * beard_rgb + (1 - beard_alpha) * roi).astype(np.uint8)
    face[y_offset:face.shape[0], 0:face_width] = blended
    return face

def transform_face(face, target_age="old", alpha=1.0, beard=True):
    if target_age == "old":
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        input_tensor = preprocess(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = generator(input_tensor)
        output_image = tensor_to_image(output_tensor)
        generated_face = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
        generated_face = cv2.resize(generated_face, (face.shape[1], face.shape[0]))
        blended = cv2.addWeighted(face, 1 - alpha, generated_face, alpha, 0)
        if beard:
            blended = add_beard(blended, alpha)
        return blended
    elif target_age == "young":
        smoothed = cv2.bilateralFilter(face, d=9, sigmaColor=75, sigmaSpace=75)
        return cv2.addWeighted(face, 0.5, smoothed, 0.5, 0)
    return face

def apply_age_model(frame, target_age="old", alpha=1.0, beard=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60,60))
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        try:
            transformed = transform_face(face_roi, target_age=target_age, alpha=alpha, beard=beard)
            frame[y:y+h, x:x+w] = transformed
        except Exception as e:
            print("Error processing face:", e)
    return frame

def main():
    try:
        base_age = int(input("Enter your current age (0-120): "))
    except:
        base_age = 30
    base_age = max(0, min(120, base_age))
    max_extra = 200
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Extra", "Controls", 0, max_extra, lambda x: None)
    cv2.createTrackbar("Beard", "Controls", 1, 1, lambda x: None)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    with pyvirtualcam.Camera(width=width, height=height, fps=30) as cam:
        print("Virtual camera started. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            extra_years = cv2.getTrackbarPos("Extra", "Controls")
            effective_age = base_age + extra_years
            effective_alpha = extra_years / max_extra
            beard_on = cv2.getTrackbarPos("Beard", "Controls") == 1
            beard_flag = extra_years > 0 and beard_on
            target_mode = "old" if extra_years > 0 else "young"
            output = apply_age_model(frame, target_age=target_mode, alpha=effective_alpha, beard=beard_flag)
            cv2.putText(output, f"Age: {effective_age}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Face Transformation Output", output)
            cam.send(output)
            cam.sleep_until_next_frame()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
