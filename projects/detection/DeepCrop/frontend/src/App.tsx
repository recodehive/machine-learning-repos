// import React from "react";
import "./App.css";
import Header from "./components/Header";
import InfoSection from "./components/InfoSection";
import Predictor from "./components/Predictor";

function App() {
  return (
    <div className="wrapper">
      <Header />
      <Predictor />  {/* moved this ABOVE info */}
      <InfoSection />  {/* moved down */}
      <footer className="footer">
        <small>© 2025 DeepCrop — AI for Sustainable Agriculture</small>
      </footer>
    </div>
  );
}

export default App;
