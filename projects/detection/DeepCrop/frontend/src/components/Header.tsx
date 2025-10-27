// import React from "react";
import "../App.css";
import logo from "../assets/logo.png";

const Header: React.FC = () => {
  return (
    <header className="header">
      <img src={logo} alt="DeepCrop Logo" className="header-logo" />
      <h1>DeepCrop</h1>
      <p className="subtitle">AI-Powered Potato Disease Detection</p>
    </header>
  );
};

export default Header;
