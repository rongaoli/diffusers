import importlib.util
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from ...utils import get_logger, load_image

logger = get_logger(__name__)

_insightface_available = importlib.util.find_spec("insightface") is not None
_consisid_eva_clip_available = importlib.util.find_spec("consisid_eva_clip") is not None
_facexlib_available = importlib.util.find_spec("facexlib") is not None

if _insightface_available:
    import insightface
    from insightface.app import FaceAnalysis
else:
    raise ImportError("insightface is not available. Please install it using 'pip install insightface'.")

if _consisid_eva_clip_available:
    from consisid_eva_clip import create_model_and_transforms
    from consisid_eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
else:
    raise ImportError("consisid_eva_clip is not available. Please install it using 'pip install consisid_eva_clip'.")

if _facexlib_available:
    from facexlib.parsing import init_parsing_model
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
else:
    raise ImportError("facexlib is not available. Please install it using 'pip install facexlib'.")

def resize_numpy_image_long(image, resize_long_edge=768):


    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image

def img2tensor(imgs, bgr2rgb=True, float32=True):


    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == "float64":
                img = img.astype("float32")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    return _totensor(imgs, bgr2rgb, float32)

def to_gray(img):

    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x

def process_face_embeddings(
    face_helper_1,
    clip_vision_model,
    face_helper_2,
    eva_transform_mean,
    eva_transform_std,
    app,
    device,
    weight_dtype,
    image,
    original_id_image=None,
    is_align_face=True,
):


    face_helper_1.clean_all()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # get antelopev2 embedding
    face_info = app.get(image_bgr)
    if len(face_info) > 0:
        face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[
            -1
        ]  # only use the maximum face
        id_ante_embedding = face_info["embedding"]  # (512,)
        face_kps = face_info["kps"]
    else:
        id_ante_embedding = None
        face_kps = None

    # using facexlib to detect and align face
    face_helper_1.read_image(image_bgr)
    face_helper_1.get_face_landmarks_5(only_center_face=True)
    if face_kps is None:
        face_kps = face_helper_1.all_landmarks_5[0]
    face_helper_1.align_warp_face()
    if len(face_helper_1.cropped_faces) == 0:
        raise RuntimeError("facexlib align face fail")
    align_face = face_helper_1.cropped_faces[0]  # (512, 512, 3)  # RGB

    # in case insightface didn't detect face
    if id_ante_embedding is None:
        logger.warning("Failed to detect face using insightface. Extracting embedding with align face")
        id_ante_embedding = face_helper_2.get_feat(align_face)

    id_ante_embedding = torch.from_numpy(id_ante_embedding).to(device, weight_dtype)  # torch.Size([512])
    if id_ante_embedding.ndim == 1:
        id_ante_embedding = id_ante_embedding.unsqueeze(0)  # torch.Size([1, 512])

    # parsing
    if is_align_face:
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0  # torch.Size([1, 3, 512, 512])
        input = input.to(device)
        parsing_out = face_helper_1.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)  # torch.Size([1, 1, 512, 512])
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)  # torch.Size([1, 3, 512, 512])
        # only keep the face features
        return_face_features_image = torch.where(bg, white_image, to_gray(input))  # torch.Size([1, 3, 512, 512])
        return_face_features_image_2 = torch.where(bg, white_image, input)  # torch.Size([1, 3, 512, 512])
    else:
        original_image_bgr = cv2.cvtColor(original_id_image, cv2.COLOR_RGB2BGR)
        input = img2tensor(original_image_bgr, bgr2rgb=True).unsqueeze(0) / 255.0  # torch.Size([1, 3, 512, 512])
        input = input.to(device)
        return_face_features_image = return_face_features_image_2 = input

    # transform img before sending to eva-clip-vit
    face_features_image = resize(
        return_face_features_image, clip_vision_model.image_size, InterpolationMode.BICUBIC
    )  # torch.Size([1, 3, 336, 336])
    face_features_image = normalize(face_features_image, eva_transform_mean, eva_transform_std)
    id_cond_vit, id_vit_hidden = clip_vision_model(
        face_features_image.to(weight_dtype), return_all_features=False, return_hidden=True, shuffle=False
    )  # torch.Size([1, 768]),  list(torch.Size([1, 577, 1024]))
    id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
    id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

    id_cond = torch.cat(
        [id_ante_embedding, id_cond_vit], dim=-1
    )  # torch.Size([1, 512]), torch.Size([1, 768])  ->  torch.Size([1, 1280])

    return (
        id_cond,
        id_vit_hidden,
        return_face_features_image_2,
        face_kps,
    )  # torch.Size([1, 1280]), list(torch.Size([1, 577, 1024]))

def process_face_embeddings_infer(
    face_helper_1,
    clip_vision_model,
    face_helper_2,
    eva_transform_mean,
    eva_transform_std,
    app,
    device,
    weight_dtype,
    img_file_path,
    is_align_face=True,
):


    # Load and preprocess the input image
    if isinstance(img_file_path, str):
        image = np.array(load_image(image=img_file_path).convert("RGB"))
    else:
        image = np.array(ImageOps.exif_transpose(Image.fromarray(img_file_path)).convert("RGB"))

    # Resize image to ensure the longer side is 1024 pixels
    image = resize_numpy_image_long(image, 1024)
    original_id_image = image

    # Process the image to extract face embeddings and related features
    id_cond, id_vit_hidden, align_crop_face_image, face_kps = process_face_embeddings(
        face_helper_1,
        clip_vision_model,
        face_helper_2,
        eva_transform_mean,
        eva_transform_std,
        app,
        device,
        weight_dtype,
        image,
        original_id_image,
        is_align_face,
    )

    # Convert the aligned cropped face image (torch tensor) to a numpy array
    tensor = align_crop_face_image.cpu().detach()
    tensor = tensor.squeeze()
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy() * 255
    tensor = tensor.astype(np.uint8)
    image = ImageOps.exif_transpose(Image.fromarray(tensor))

    return id_cond, id_vit_hidden, image, face_kps

def prepare_face_models(model_path, device, dtype):
    Prepare all face models for the facial recognition task.
    - model_path: Path to the directory containing model files.
    - device: The device (e.g., 'cuda', 'xpu', 'cpu') where models will be loaded.
    - dtype: Data type (e.g., torch.float32) for model inference.
