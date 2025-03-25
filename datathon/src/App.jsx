import axios from "axios";
import { Sidebar } from "./components/Sidebar";
import { Prediction } from "./components/Prediction";
import { Analysis } from "./components/Analysis";
import { useState, useEffect } from "react";

export const App = () => {
  const [selected, setSelected] = useState("prediction");
  const [isLoadingData, setIsLoadingData] = useState(true);
  
  const [defaultImages, setDefaultImages] = useState({
    topClients: null,
    topArticles: null,
  });

  useEffect(() => {
    
    const fetchImages = async () => {
      try {
        setIsLoadingData(true);
  
        const [clientsRes, articlesRes] = await Promise.all([
          axios.get("http://127.0.0.1:5000/top-clientes", { responseType: "blob" }),
          axios.get("http://127.0.0.1:5000/top-articulos", { responseType: "blob" }),
        ]);
  
        // Clean up old image URLs before setting new ones
        setDefaultImages((prev) => {
          if (prev.topClients) URL.revokeObjectURL(prev.topClients);
          if (prev.topArticles) URL.revokeObjectURL(prev.topArticles);
          
          return {
            topClients: URL.createObjectURL(clientsRes.data),
            topArticles: URL.createObjectURL(articlesRes.data),
          };
        });
        
      } catch (error) {
        console.error("Error fetching images:", error);
      } finally {
        setIsLoadingData(false);
      }
    };
  
    fetchImages();
  }, []);
  

  return (
    <div style={{ display: "flex" }}>
      <div style={{ flex: 1 }}>
        <Sidebar setSelected={setSelected} selected={selected} />
      </div>
      <div style={{ flex: 3 }}>
        {selected === "prediction" ? (
          <Prediction />
        ) : (
          <Analysis 
            defaultImages={defaultImages} 
            isLoadingData={isLoadingData} 
          />
        )}
      </div>
    </div>
  );
};
