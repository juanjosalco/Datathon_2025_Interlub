import Select from "react-select";
import analysisLogo from "../images/predictive-analysis.png";
import { useState, useEffect } from "react";
import axios from "axios";
import datathonJSON from "../utils/DatathonJSON.json";

import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export function processData(isEmpty = false) {
  if (isEmpty) {
    return {
      labels: [],
      datasets: [
        {
          label: "No hay datos",
          data: [],
          borderColor: "rgba(200, 200, 200, 0.5)",
          backgroundColor: "rgba(200, 200, 200, 0.2)",
        },
      ],
    };
  }

  return {
    labels: ["A", "B", "C", "D", "E"],
    datasets: [
      {
        label: "Predicción de Ventas del artículo",
        data: [65, 59, 80, 81, 56],
        fill: false,
        borderColor: "rgba(191, 49, 49, 1)",
        backgroundColor: "rgba(191, 49, 49, 1)",
        tension: 0.5,
      },
    ],
  };
}

export const Prediction = () => {
  const [selectedArticle, setSelectedArticle] = useState(null);
  const [selectedTime, setSelectedTime] = useState(null);
  const [isAlredyPredicted, setIsAlreadyPredicted] = useState(false);
  const [hovered, setHovered] = useState(null);
  const [timeOptions, setTimeOptions] = useState([]);
  const [predict, setPredict] = useState(0.0)
  const [imageSrc, setImageSrc] = useState(null); // State for the fetched image
  const [isLoading, setIsLoading] = useState(false); // ✅ New state for loading



  const fetchPredictionFilter = async (articleId) => {
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/prediction_filter",
        { id: articleId.value },
      );
      const weeks = response.data.weeks;
      console.log(weeks)
      if (weeks.length > 2) {
        setTimeOptions([
          {
            value: 1,
            label: "1 Semanas",
          },
          {
            value: 2,
            label: "2 Semanas",
          },
          {
            value: 3,
            label: "3 Semanas",
          },
        ]);
      } else if (weeks.length > 1) {
        setTimeOptions([
          {
            value: 1,
            label: "1 Semanas",
          },
          {
            value: 2,
            label: "2 Semanas",
          },
        ]);
      } else if (weeks.length > 0) {
        setTimeOptions([
          {
            value: 1,
            label: "1 Semanas",
          },
        ]);
      }

      return response.data;
    } catch (error) {
      console.error("Error fetching prediction filter:", error);
    }
  };

  const fetchPrediction = async (articleId, time) => {
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predictions",
        { id: articleId.value, time: time.value }
      );
      
      setPredict(response.data.prediction)

      return response.data;
    } catch (error) {
      console.error("Error fetching prediction filter:", error);
    }
  }

  const fetchImage = async (articleId, time) => {
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predictions/images",
        { id: articleId.value, time: time.value },
        {responseType: 'blob'}
      );

      // Create a local URL for the image blob
      const imageUrl = URL.createObjectURL(response.data);
      setImageSrc(imageUrl); // Update state with the image URL

      return response.data;
    } catch (error) {
      console.error("Error fetching prediction filter:", error);
    }
  }

  function getArticlesOptions() {
    const eliminateArticleDuplicates = datathonJSON.filter(
      (article, index, self) =>
        index === self.findIndex((t) => t.Articulo === article.Articulo)
    );
    return eliminateArticleDuplicates.map((article) => ({
      value: article.Articulo,
      label: article.Articulo,
    }));
  }

  const articleOptions = getArticlesOptions();

  const data = processData(!isAlredyPredicted);

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text:
          selectedArticle !== null
            ? `Predicción de Ventas del artículo ${selectedArticle?.label}`
            : "Elija un artículo para ver la predicción",
      },
    },
  };

  useEffect(() => {
    if (selectedArticle) {
      fetchPredictionFilter(selectedArticle);
    }
  }, [selectedArticle]);

  return (
    <div style={{marginBottom: 32}}>
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          justifyContent: "space-between",
          marginInline: 56,
          marginTop: 56,
        }}
      >
        <Select
          styles={customStyles}
          placeholder="Seleccionar Articulo"
          value={selectedArticle} // Use value instead of defaultValue
          onChange={setSelectedArticle}
          options={articleOptions}
        />
        <Select
          isDisabled={selectedArticle === null}
          styles={customStyles}
          placeholder="Seleccionar Periodo"
          value={selectedTime} // Use value instead of defaultValue
          onChange={setSelectedTime}
          options={timeOptions}
        />
        <div
          onClick={() => {
            if (selectedArticle && selectedTime) {
              fetchImage(selectedArticle, selectedTime);
              fetchPrediction(selectedArticle, selectedTime);
            } else {
              alert("Por favor selecciona un artículo y un periodo.");
            }
          }}
          style={{
            ...styles.predictionButton,
            backgroundColor: hovered === "calculate" ? "#BF3131" : "#7D0A0A",
          }}
          onMouseEnter={() => setHovered("calculate")}
          onMouseLeave={() => setHovered(null)}
        >
          <img
            src={analysisLogo}
            style={{ width: 32, height: 32 }}
            alt="logo"
          />
          <p style={{ color: "#FFFFFF" }}>Calcular Predicción</p>
        </div>
      </div>
      {imageSrc !== null ? <img src={imageSrc} /> : <Line style={{ marginInline: 56 }} data={data} options={options} />}
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          justifyContent: "space-between",
          marginInline: 56,
          marginTop: 32,
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
          }}
        >
          <div
            style={{
              width: 16,
              height: 16,
              backgroundColor: "rgba(191, 49, 49, 1)",
              borderRadius: 4,
              marginRight: 8,
            }}
          ></div>
          <p>Predicción de Ventas: </p>
          <div
            style={{
              marginLeft: 16,
              border: "1px solid #000",
              padding: 8,
              width: 208,
              borderRadius: 8,
            }}
          >
            <p>{predict}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

const customStyles = {
  control: (provided) => {
    return {
      ...provided,
      minWidth: 208,
      display: "flex",
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "space-between",
      height: 40,
      backgroundColor: "#FFFFFF",
      borderRadius: 8,
      marginBottom: 32,
    };
  },
};

const styles = {
  predictionButton: {
    display: "flex",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-evenly",
    width: 200,
    height: 40,
    borderRadius: 8,
    marginBottom: 32,
    cursor: "pointer",
  },
};
