from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Tuple, Annotated
import cv2
import numpy as np
import base64
import torch
from io import BytesIO
from PIL import Image
from unetmodel import UNet as secondaryUNet
from model import UNet as originalUNet
from metrics import calcAreaandCentroid
import matplotlib.pyplot as plt

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scaleValueMap = {
    "10": 0.10394100844860077,
    "30": 0.10394100844860077,
    "50": 0.09130698442459106,
    "70": 0.1150854229927063,
    "200": 0.102478988468647,
    "500": 0.09072770178318024,
    "1000": 0.11125108599662781,
    "2000": 0.11125108599662781,
    "3000": 0.10487890243530273,
    "5000": 0.07409382611513138,
    "10000": 0.0821211189031601,
    "20000": 0.07376280426979065,
    "30000": 0.0584530234336853,
    "40000": 0.0623149499297142,
    "50000": 0.12603674829006195,
    "100000": 0.06328043341636658
}

class ProcessedData(BaseModel):
    area: float
    centroid: Tuple[float, float]
    image: str

def segment_image_and_calculate_values(image, scale, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not model_path.__contains__('X'):
        model = originalUNet().to(device)
    else:
        model = secondaryUNet().to(device)

    # Load the model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = torch.from_numpy(image).permute(2, 0, 1).float()
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        outputs = model(image)
        outputs = torch.where(outputs > 0.5, 255, 0)
        
    grayMask = cv2.cvtColor(np.float32(np.transpose(outputs[0].cpu().numpy(), (1, 2, 0))), cv2.COLOR_BGR2GRAY)
    grayMask = np.uint8(grayMask)
    calcArea, calcCentX, calcCentY = calcAreaandCentroid(grayMask)

    calcArea = calcArea * (scale ** 2)
    return grayMask, calcArea, calcCentX, calcCentY

def encode_image_to_base64(image: np.ndarray) -> str:
    image = Image.fromarray(image)
    buff = BytesIO()
    image.save(buff, format="PNG")
    imageBytes = buff.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(imageBytes)
    return im_b64


@app.post("/api/calculate", response_model=ProcessedData)
async def calculate(file: UploadFile = File(...), scale: Annotated[int, Form()] = 1, model: Annotated[str, Form()] = './modelweights/torchmodelXcentroid2.pth'):
    mapValue = scaleValueMap[str(scale)]
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    img = np.asarray(img)
    mask, area, centroidx, centroidy = segment_image_and_calculate_values(img, scale, model)

    # Scale the area and centroid values
    scaled_area = area * (mapValue ** 2)
    scaled_centroid = (centroidx, centroidy)

    # Optionally, draw the centroid on the image for visualization
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.circle(mask, (int(centroidx), int(centroidy)), 3, (255, 0, 0), -1)

    img_base64 = encode_image_to_base64(mask)
    return ProcessedData(area=scaled_area, centroid=scaled_centroid, image=img_base64)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)