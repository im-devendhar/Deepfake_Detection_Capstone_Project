from django.shortcuts import render, redirect
import os
import time
import glob
import shutil
import base64
import json
from io import BytesIO
from django.conf import settings
from .forms import VideoUploadForm

try:
    import torch
    from torchvision import transforms, models
    from torch.utils.data.dataset import Dataset
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from torch import nn
    from PIL import Image as pImage
    ML_DEPENDENCIES_AVAILABLE = True
    ML_DEPENDENCY_ERROR = ""
except ImportError as exc:
    torch = transforms = models = Dataset = np = cv2 = plt = nn = pImage = None
    ML_DEPENDENCIES_AVAILABLE = False
    ML_DEPENDENCY_ERROR = str(exc)

index_template_name = 'index.html'
predict_template_name = 'predict.html'
about_template_name = "about.html"

im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
SUPPORTED_SEQUENCE_LENGTHS = {20, 40, 60, 80, 100}
if ML_DEPENDENCIES_AVAILABLE:
    sm = nn.Softmax(dim=1)
    inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        face_cascades = [
            cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name)
            for cascade_name in (
                'haarcascade_frontalface_default.xml',
                'haarcascade_frontalface_alt2.xml',
                'haarcascade_profileface.xml',
            )
        ]
    else:
        face_cascades = []

    train_transforms = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((im_size,im_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean,std)])

    def read_video_frames(video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def select_evenly_spaced_frames(frames, count):
        if not frames:
            return []
        non_blank_frames = [
            frame for frame in frames
            if frame is not None and np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > 3
        ]
        if non_blank_frames:
            frames = non_blank_frames
        if len(frames) >= count:
            indices = np.linspace(0, len(frames) - 1, count, dtype=int)
            return [frames[index] for index in indices]

        selected = list(frames)
        selected.extend([frames[-1]] * (count - len(selected)))
        return selected

    def _detect_faces(gray):
        if not face_cascades:
            return []

        equalized_gray = cv2.equalizeHist(gray)
        detected_faces = []

        for cascade in face_cascades:
            if cascade.empty():
                continue

            faces = cascade.detectMultiScale(
                equalized_gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(35, 35),
            )
            detected_faces.extend(faces)

        flipped_gray = cv2.flip(equalized_gray, 1)
        frame_width = gray.shape[1]
        for cascade in face_cascades:
            if cascade.empty():
                continue

            faces = cascade.detectMultiScale(
                flipped_gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(35, 35),
            )
            for x, y, w, h in faces:
                detected_faces.append((frame_width - x - w, y, w, h))

        return detected_faces

    def crop_center_frame(frame, crop_ratio=0.55):
        frame_height, frame_width = frame.shape[:2]
        side = int(min(frame_height, frame_width) * crop_ratio)
        center_x = frame_width // 2
        center_y = int(frame_height * 0.42)
        left = max(0, min(center_x - side // 2, frame_width - side))
        top = max(0, min(center_y - side // 2, frame_height - side))
        return frame[top:top + side, left:left + side]

    def crop_largest_face(frame, margin=0.35, fallback_to_frame=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _detect_faces(gray)
        if len(faces) == 0:
            return (crop_center_frame(frame), False) if fallback_to_frame else (None, False)

        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        frame_height, frame_width = frame.shape[:2]
        center_x = x + (w / 2)
        center_y = y + (h / 2)
        side = int(max(w, h) * (1 + margin * 2))

        left = int(round(center_x - side / 2))
        top = int(round(center_y - side / 2))
        left = max(0, min(left, frame_width - side))
        top = max(0, min(top, frame_height - side))
        right = min(left + side, frame_width)
        bottom = min(top + side, frame_height)

        return frame[top:bottom, left:right], True

    def decode_client_face_frames(encoded_frames):
        frames = []
        if not encoded_frames:
            return frames

        try:
            frame_data = json.loads(encoded_frames)
        except json.JSONDecodeError:
            return frames

        for data_url in frame_data:
            if not isinstance(data_url, str) or "," not in data_url:
                continue
            try:
                _, encoded_image = data_url.split(",", 1)
                image_bytes = base64.b64decode(encoded_image)
                image = pImage.open(BytesIO(image_bytes)).convert("RGB")
                frames.append(np.array(image))
            except Exception:
                continue

        return frames

    def build_prediction_tensor(frames, sequence_length):
        sampled_frames = select_evenly_spaced_frames(frames, sequence_length)
        if not sampled_frames:
            raise ValueError("Could not read usable frames for prediction.")
        transformed_frames = [
            train_transforms(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            for frame in sampled_frames
        ]
        return torch.stack(transformed_frames).unsqueeze(0)

    class Model(nn.Module):

        def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
            super(Model, self).__init__()
            model = models.resnext50_32x4d(pretrained = False)
            self.model = nn.Sequential(*list(model.children())[:-2])
            self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
            self.relu = nn.LeakyReLU()
            self.dp = nn.Dropout(0.4)
            self.linear1 = nn.Linear(2048,num_classes)
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            batch_size,seq_length, c, h, w = x.shape
            x = x.view(batch_size * seq_length, c, h, w)
            fmap = self.model(x)
            x = self.avgpool(fmap)
            x = x.view(batch_size,seq_length,2048)
            x_lstm,_ = self.lstm(x,None)
            return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))
          
 


    class validation_dataset(Dataset):
        def __init__(self,video_names,sequence_length=60,transform = None):
            self.video_names = video_names
            self.transform = transform
            self.count = sequence_length

        def __len__(self):
            return len(self.video_names)

        def __getitem__(self,idx):
            video_path = self.video_names[idx]
            video_frames = read_video_frames(video_path)
            if not video_frames:
                raise ValueError("Could not read frames from the uploaded video.")

            sampled_frames = select_evenly_spaced_frames(video_frames, self.count)
            frames = []
            for frame in sampled_frames:
                face_frame, _ = crop_largest_face(frame)
                frames.append(self.transform(face_frame))

            return torch.stack(frames).unsqueeze(0)
else:
    sm = inv_normalize = device = train_transforms = face_cascades = None
    read_video_frames = select_evenly_spaced_frames = crop_largest_face = None
    decode_client_face_frames = build_prediction_tensor = None
    Model = validation_dataset = None

def im_convert(tensor, video_file_name):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    # This image is not used
    # cv2.imwrite(os.path.join(settings.PROJECT_DIR, 'uploaded_images', video_file_name+'_convert_2.png'),image*255)
    return image

def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1,2,0)
    b,g,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    image = image*[0.22803, 0.22145, 0.216989] +  [0.43216, 0.394666, 0.37645]
    image = image*255.0
    plt.imshow(image.astype('uint8'))
    plt.show()


def predict(model,img,path = './', video_file_name=""):
  fmap,logits = model(img.to(device))
  img = im_convert(img[:,-1,:,:,:], video_file_name)
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _,prediction = torch.max(logits,1)
  confidence = logits[:,int(prediction.item())].item()*100
  print('confidence of prediction:',logits[:,int(prediction.item())].item()*100)  
  return [int(prediction.item()),confidence]

def plot_heat_map(i, model, img, path = './', video_file_name=''):
  fmap,logits = model(img.to(device))
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _,prediction = torch.max(logits,1)
  idx = np.argmax(logits.detach().cpu().numpy())
  bz, nc, h, w = fmap.shape
  #out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  out = np.dot(fmap[i].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  predict = out.reshape(h,w)
  predict = predict - np.min(predict)
  predict_img = predict / np.max(predict)
  predict_img = np.uint8(255*predict_img)
  out = cv2.resize(predict_img, (im_size,im_size))
  heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
  img = im_convert(img[:,-1,:,:,:], video_file_name)
  result = heatmap * 0.5 + img*0.8*255
  # Saving heatmap - Start
  heatmap_name = video_file_name+"_heatmap_"+str(i)+".png"
  image_name = os.path.join(settings.PROJECT_DIR, 'uploaded_images', heatmap_name)
  cv2.imwrite(image_name,result)
  # Saving heatmap - End
  result1 = heatmap * 0.5/255 + img*0.8
  r,g,b = cv2.split(result1)
  result1 = cv2.merge((r,g,b))
  return image_name

# Model Selection
def get_accurate_model(sequence_length):
    if sequence_length not in SUPPORTED_SEQUENCE_LENGTHS:
        return ""

    final_model = ""
    sequence_models = []
    list_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "*.pt"))

    for model_path in list_models:
        model_filename = os.path.basename(model_path)
        try:
            accuracy = float(model_filename.split("_")[1])
            seq = model_filename.split("_")[3]
            if int(seq) == sequence_length:
                sequence_models.append((accuracy, model_path))
        except (IndexError, ValueError):
            pass  # Handle cases where the filename format doesn't match expected

    if sequence_models:
        final_model = max(sequence_models, key=lambda item: item[0])[1]
    else:
        print("No model found for the specified sequence length.")  # Handle no models found case

    return final_model

ALLOWED_VIDEO_EXTENSIONS = set(['mp4','gif','webm','avi','3gp','wmv','flv','mkv'])

def allowed_video_file(filename):
    if '.' not in filename:
        return False
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        if 'file_name' in request.session:
            del request.session['file_name']
        if 'preprocessed_images' in request.session:
            del request.session['preprocessed_images']
        if 'faces_cropped_images' in request.session:
            del request.session['faces_cropped_images']
        if 'client_face_frames' in request.session:
            del request.session['client_face_frames']
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = video_file.name.split('.')[-1]
            sequence_length = video_upload_form.cleaned_data['sequence_length']
            video_content_type = video_file.content_type.split('/')[0]
            if video_content_type in settings.CONTENT_TYPES:
                if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                    video_upload_form.add_error("upload_video_file", "Maximum file size 100 MB")
                    return render(request, index_template_name, {"form": video_upload_form})

            if sequence_length <= 0:
                video_upload_form.add_error("sequence_length", "Sequence Length must be greater than 0")
                return render(request, index_template_name, {"form": video_upload_form})

            if sequence_length not in SUPPORTED_SEQUENCE_LENGTHS:
                video_upload_form.add_error("sequence_length", "Choose one of the supported frame depths: 20, 40, 60, 80, or 100.")
                return render(request, index_template_name, {"form": video_upload_form})
            
            if allowed_video_file(video_file.name) == False:
                video_upload_form.add_error("upload_video_file","Only video files are allowed ")
                return render(request, index_template_name, {"form": video_upload_form})
            
            saved_video_file = 'uploaded_file_'+str(int(time.time()))+"."+video_file_ext
            if settings.DEBUG:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file)
            else:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file)
            request.session['sequence_length'] = sequence_length
            request.session['client_face_frames'] = request.POST.get('client_face_frames', '')
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})

def predict_page(request):
    if request.method == "GET":
        if not ML_DEPENDENCIES_AVAILABLE:
            return render(request, 'cuda_full.html', {
                "error_title": "Prediction dependencies are not installed",
                "error_message": ML_DEPENDENCY_ERROR,
            })
        # Redirect to 'home' if 'file_name' is not in session
        if 'file_name' not in request.session:
            return redirect("ml_app:home")
        if 'file_name' in request.session:
            video_file = request.session['file_name']
        if 'sequence_length' in request.session:
            sequence_length = request.session['sequence_length']
        client_face_frames = decode_client_face_frames(request.session.get('client_face_frames', ''))
        path_to_videos = [video_file]
        video_file_name = os.path.basename(video_file)
        video_file_name_only = os.path.splitext(video_file_name)[0]
        # Production environment adjustments
        if not settings.DEBUG:
            production_video_name = os.path.join('/home/app/staticfiles/', video_file_name )
            print("Production file name", production_video_name)
        else:
            production_video_name = video_file_name

        if client_face_frames:
            prediction_tensor = build_prediction_tensor(client_face_frames, sequence_length)
        else:
            prediction_tensor = None
            video_dataset = validation_dataset(path_to_videos, sequence_length=sequence_length, transform=train_transforms)

        # Load model
        model = Model(2).to(device)  # Adjust the model instantiation according to your model structure
        path_to_model = get_accurate_model(sequence_length)
        if not path_to_model:
            return render(request, 'cuda_full.html', {
                "error_title": "No model found",
                "error_message": f"No trained model is available for {sequence_length} frames.",
            })
        model.load_state_dict(torch.load(path_to_model, map_location=device))
        model.eval()
        start_time = time.time()
        # Display preprocessing images
        print("<=== | Started Videos Splitting | ===>")
        preprocessed_images = []
        faces_cropped_images = []
        frames = client_face_frames if client_face_frames else read_video_frames(video_file)

        print(f"Number of frames: {len(frames)}")
        # Process each frame for preprocessing and face cropping
        faces_found = 0
        sampled_frames = select_evenly_spaced_frames(frames, sequence_length)
        for i, frame in enumerate(sampled_frames):

            if client_face_frames:
                rgb_frame = frame
                frame_face = frame
                face_found = True
            else:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_face, face_found = crop_largest_face(frame)

            # Save preprocessed image
            image_name = f"{video_file_name_only}_preprocessed_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            img_rgb = pImage.fromarray(rgb_frame, 'RGB')
            img_rgb.save(image_path)
            preprocessed_images.append(image_name)

            # Convert cropped face image to RGB and save. If OpenCV cannot detect
            # a face, this is a centered fallback crop so prediction can continue.
            rgb_face = frame_face if client_face_frames else cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
            img_face_rgb = pImage.fromarray(rgb_face, 'RGB')
            image_name = f"{video_file_name_only}_cropped_faces_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            img_face_rgb.save(image_path)
            faces_cropped_images.append(image_name)
            if face_found:
                faces_found += 1

        print("<=== | Videos Splitting and Face Cropping Done | ===>")
        print("--- %s seconds ---" % (time.time() - start_time))

        if faces_found == 0:
            print("No faces detected by OpenCV. Using centered fallback crops.")

        # Perform prediction
        try:
            heatmap_images = []
            output = ""
            confidence = 0.0

            for i in range(len(path_to_videos)):
                print("<=== | Started Prediction | ===>")
                with torch.no_grad():
                    model_input = prediction_tensor if prediction_tensor is not None else video_dataset[i]
                    prediction = predict(model, model_input, './', video_file_name_only)
                confidence = round(prediction[1], 1)
                output = "REAL" if prediction[0] == 1 else "FAKE"
                print("Prediction:", prediction[0], "==", output, "Confidence:", confidence)
                print("<=== | Prediction Done | ===>")
                print("--- %s seconds ---" % (time.time() - start_time))

                # Uncomment if you want to create heat map images
                # for j in range(sequence_length):
                #     heatmap_images.append(plot_heat_map(j, model, video_dataset[i], './', video_file_name_only))

            # Render results
            context = {
                'preprocessed_images': preprocessed_images,
                'faces_cropped_images': faces_cropped_images,
                'heatmap_images': heatmap_images,
                'original_video': production_video_name,
                'models_location': os.path.join(settings.PROJECT_DIR, 'models'),
                'model_name': os.path.basename(path_to_model),
                'sequence_length': sequence_length,
                'processing_time': round(time.time() - start_time, 1),
                'output': output,
                'confidence': confidence
            }

            if settings.DEBUG:
                return render(request, predict_template_name, context)
            else:
                return render(request, predict_template_name, context)

        except Exception as e:
            print(f"Exception occurred during prediction: {e}")
            return render(request, 'cuda_full.html', {
                "error_title": "Prediction failed",
                "error_message": str(e),
            })
def about(request):
    return render(request, about_template_name)

def handler404(request,exception):
    return render(request, '404.html', status=404)
def cuda_full(request):
    return render(request, 'cuda_full.html')
