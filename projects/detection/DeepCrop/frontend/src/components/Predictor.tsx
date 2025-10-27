import { useState } from "react";
import axios from "axios";
import "../App.css";

const Predictor = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("data", file); // 'data' is required by Gradio API

    try {
      const res = await axios.post(
        "https://farhan2127-deepcrop.hf.space/run/predict",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      const label = res.data?.data?.[0];
      setResult(label || "Unknown");
    } catch (err) {
      console.error(err);
      alert("Prediction failed. Make sure the Hugging Face Space is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="predictor">
      <h2>Try It Yourself</h2>
      <p>Upload a potato leaf image and let DeepCrop detect the disease.</p>

      <label className="upload">
        <input type="file" accept="image/*" onChange={handleFile} />
        Choose Image
      </label>

      <button onClick={handleSubmit} className="btn" disabled={!file || loading}>
        {loading ? "Predicting..." : "Predict"}
      </button>

      {preview && <img src={preview} alt="preview" className="preview" />}
      {result && <h3 className="prediction">Prediction: {result}</h3>}

      {/* 👇 New Section: Link to Hugging Face Demo */}
      <div style={{ marginTop: "20px" }}>
        <a
          href="https://huggingface.co/spaces/farhan2127/DeepCrop"
          target="_blank"
          rel="noopener noreferrer"
          className="btn"
          style={{
            backgroundColor: "#2e7d32",
            color: "white",
            padding: "10px 18px",
            borderRadius: "8px",
            textDecoration: "none",
            display: "inline-block",
            transition: "background 0.3s",
          }}
          onMouseOver={(e) => {
            (e.target as HTMLElement).style.backgroundColor = "#1b5e20";
          }}
          onMouseOut={(e) => {
            (e.target as HTMLElement).style.backgroundColor = "#2e7d32";
          }}
        >
          🌿 Open Full Demo on Hugging Face
        </a>
      </div>
    </section>
  );
};

export default Predictor;
