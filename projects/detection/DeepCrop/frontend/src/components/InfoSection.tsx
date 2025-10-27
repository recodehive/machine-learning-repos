// import React from "react";
import "../App.css";
import healthy from "../assets/healthy.jpeg";
import early from "../assets/early_blight.jpeg";
import late from "../assets/late_blight.jpeg";

const InfoSection: React.FC = () => {
  const info = [
    {
      title: "Healthy Leaf",
      img: healthy,
      desc: "A vibrant green potato leaf with no visible spots or discoloration, indicating strong plant health.",
    },
    {
      title: "Early Blight",
      img: early,
      desc: "Dark brown circular spots with concentric rings appear on older leaves. Caused by Alternaria solani fungus.",
    },
    {
      title: "Late Blight",
      img: late,
      desc: "Large irregular dark lesions on leaves and stems, often accompanied by white mold on the undersides.",
    },
  ];

  return (
    <section className="info-section">
      <h2>Potato Leaf Conditions</h2>
      <div className="info-grid">
        {info.map((item, i) => (
          <div key={i} className="info-card">
            <img src={item.img} alt={item.title} className="info-img" />
            <h3>{item.title}</h3>
            <p>{item.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
};

export default InfoSection;
