import React, { useState, useRef } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [scale, setScale] = useState("1");
  const [area, setArea] = useState(null);
  const [centroid, setCentroid] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const [isFetching, setIsFetching] = useState(false);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();

    // Validate file type (only images)
    if (!file || !file.type.startsWith("image/")) {
      alert("Please upload a valid image file.");
      return;
    }

    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        setImage(file); // Store the file for submission
      };
      img.src = e.target.result;
    };

    reader.readAsDataURL(file);
  };

  const handleScaleChange = (event) => {
    setScale(event.target.value);
  };

  const handleSubmit = (e, model) => {
    if (!image) {
      alert("Please upload an image first.");
      return;
    }
    fetchAreaAndCentroid(model);
  };

  const fetchAreaAndCentroid = (model) => {
    const formData = new FormData();
    formData.append("file", image);
    formData.append("scale", scale);
    formData.append("model", model);

    setIsFetching(true);
    fetch("http://localhost:8000/api/calculate", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        setArea(data.area);
        setCentroid(data.centroid);
        console.log(data.image);
        setProcessedImage(data.image);
        setIsFetching(false);
      })
      .catch((error) => {
        console.error("Error fetching area and centroid:", error);
      });
  };

  return (
    <div className="flex justify-center items-center h-screen">
      <div className="flex flex-col items-center w-full max-w-screen-lg p-6 rounded-lg bg-gray-100 border border-black">
        <h1 className="text-3xl font-bold mb-8 text-center">
          Glacial Lake Delineation
        </h1>
        <div className="flex flex-col items-center">
          <div className="mb-4">
            <h2 className="text-xl mb-2">Upload Image</h2>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="mb-2"
            />
            {image && (
              <div className="container">
                {" "}
                {/* Wrap image for centering */}
                <img
                  src={URL.createObjectURL(image)}
                  alt="Uploaded"
                  className="max-w-full h-auto mb-4"
                />
              </div>
            )}
          </div>
          <div className="mb-4 w-full">
            <h2 className="text-xl mb-2">Scale</h2>
            <select
              value={scale}
              onChange={handleScaleChange}
              className="p-2 border rounded w-full"
            >
              {[
                "1",
                "10",
                "30",
                "50",
                "100",
                "200",
                "300",
                "400",
                "500",
                "600",
                "700",
                "800",
                "900",
                "1000",
                "2000",
                "3000",
                "4000",
                "5000",
                "10000",
                "30000",
                "50000",
                "100000",
              ].map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={(e) =>
              handleSubmit(e, "./modelweights/torchmodelcentroid36.pth")
            }
            disabled={isFetching}
            onContextMenu={(e) =>
              e.preventDefault() ||
              handleSubmit(e, "./modelweights/torchmodelXcentroid2.pth")
            }
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition duration-300 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            Submit
          </button>
        </div>
        {processedImage && (
          <div className="mt-8 justify-center">
            <h2 className="text-xl mb-4 font-medium">Results</h2>
            {area !== null && (
              <p className="text-lg">Area: {area.toFixed(2)} units²</p>
            )}
            {centroid && (
              <p className="text-lg">
                Centroid: (X: {centroid[0].toFixed(2)}, Y:{" "}
                {centroid[1].toFixed(2)}) units
              </p>
            )}
            <div className="mt-4">
              <h3 className="text-lg mb-2">Processed Image:</h3>
              <img
                src={`data:image/png;base64,${processedImage}`}
                alt="Processed"
                className="max-w-full h-auto"
                style={{ maxWidth: "400px" }}
              />
            </div>
          </div>
        )}
        <canvas ref={canvasRef} className="hidden" />
      </div>
    </div>
  );
}

export default App;
