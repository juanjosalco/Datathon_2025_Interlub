import Select from "react-select";
import analysisLogo from "../images/predictive-analysis.png";
import { useState, useEffect } from "react";

import default1 from '../images/default1.png'
import default2 from '../images/default2.png'
import default3 from '../images/default3.png'

import articulo_mes from '../images/articulo-mes.png'
import articulo from '../images/articulo.png'

import client from '../images/cliente.png'
import client2 from '../images/cliente2.png'

import segmento from '../images/segmento.png'
import segmento2 from '../images/segmento2.png'


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

const clientOptions = [
  {value: '68F46D1A', label: '68F46D1A'},
  {value: 'IVP11137', label: 'IVP11137'},
  {value: 'IVP11576', label: 'IVP11576'},
  {value: 'IVP11031', label: 'IVP11031'},
  {value: 'IVP11066', label: 'IVP11066'},
  {value: 'IVP08053', label: 'IVP08053'},
  {value: 'IVP11141', label: 'IVP11141'},
  {value: 'IVP08016', label: 'IVP08016'},
  {value: 'B678967B', label: 'B678967B'}
]

const segmentOptions = [
  {value: 'Alimentos y farmaceutica', label: 'Alimentos y farmaceutica'},
  {value: 'Metal Mecanica', label: 'Metal Mecanica'},
  {value: 'Distribuidor', label: 'Distribuidor'},
  {value: 'Minas', label: 'Minas'},
  {value: 'Genericos', label: 'Genericos'},
  {value: 'Intercompanias', label: 'Intercompanias'},
  {value: 'Aceros', label: 'Aceros'},
]

const articlesOptions = [
  { value: "IVP09009", label: "IVP09009" },
  { value: "IVP04039", label: "IVP04039" },
  { value: "IVP07412", label: "IVP07412" },
  { value: "IVP11430", label: "IVP11430" },
  { value: "IVP11280", label: "IVP11280" },
  { value: "IVP11221", label: "IVP11221" },
  { value: "IVP11162", label: "IVP11162" },
  { value: "IVP08042", label: "IVP08042" },
];

const timeOptions = [
  { value: 1, label: "1 Semana" },
  { value: 2, label: "2 Semanas" },
  { value: 3, label: "3 Semanas" },
  { value: 4, label: "4 Semanas" },
];

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

export const Analysis = () => {
  const [selectedArticle, setSelectedArticle] = useState(null);
  const [selectedClient, setSelectedClient] = useState(null);
  const [selectedSegment, setSelectedSegment] = useState(null);
  const [displayed, setDisplayed] = useState("default")
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    setIsLoading(true);
  
    if (selectedArticle?.value === "IVP09009") {
      setTimeout(() => {
        setDisplayed("article");
        setIsLoading(false);
      }, 5000);
    } else if (selectedClient?.value === "68F46D1A") {
      setTimeout(() => {
        setDisplayed("client");
        setIsLoading(false);
      }, 5000);
    } else if (selectedSegment?.value === "Aceros") {
      setTimeout(() => {
        setDisplayed("segment");
        setIsLoading(false);
      }, 5000);
    } else {
      setTimeout(() => {
        setDisplayed("default");
        setIsLoading(false);
      }, 5000);
    }
  }, [selectedArticle, selectedClient, selectedSegment]);

  return (
    <>
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
          defaultValue={selectedArticle}
          onChange={setSelectedArticle}
          options={articlesOptions}
          isClearable
        />
        <Select
          styles={customStyles}
          placeholder="Seleccionar Cliente"
          defaultValue={selectedClient}
          onChange={setSelectedClient}
          options={clientOptions}
          isClearable
        />
        <Select
          styles={customStyles}
          placeholder="Seleccionar Segmento"
          defaultValue={selectedSegment}
          onChange={setSelectedSegment}
          options={segmentOptions}
          isClearable
        />
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          justifyContent: "space-around",
          marginInline: 16,
        }}
      >
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          justifyContent: "space-between",
          marginInline: 56,
          marginTop: 32,
        }}
      >
      </div>
      {isLoading ? (
  <div style={{ textAlign: "center", fontSize: "20px", marginTop: "20px" }}>
    Cargando...
  </div>
) : (
  displayed === "default" ? (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginInline: 56 }}>
      <div style={{ flex: 1 }}>
        <img src={default1} alt="Top Clients" style={{ width: "100%" }} />
      </div>
      <div style={{ flex: 1, marginBottom: 24 }}>
        <img src={default2} alt="Top Articles" style={{ width: "100%" }} />
      </div>
      <div style={{ flex: 1 }}>
        <img src={default3} alt="Top Consumers" style={{ width: "100%" }} />
      </div>
    </div>
  ) : displayed === "client" ? (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginInline: 56 }}>
      <div style={{ flex: 1 }}>
        <img src={client} alt="Top Clients" style={{ width: "100%" }} />
      </div>
      <div style={{ flex: 1, marginBottom: 24 }}>
        <img src={client2} alt="Top Articles" style={{ width: "100%" }} />
      </div>
    </div>
  ) : displayed === "segment" ? (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginInline: 56 }}>
      <div style={{ flex: 1 }}>
        <img src={segmento} alt="Top Clients" style={{ width: "100%" }} />
      </div>
      <div style={{ flex: 1, marginBottom: 24 }}>
        <img src={segmento2} alt="Top Articles" style={{ width: "100%" }} />
      </div>
    </div>
  ) : displayed === "article" ? (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginInline: 120 }}>
      <div style={{ flex: 1 }}>
        <img src={articulo_mes} alt="Top Clients" style={{ width: "100%" }} />
      </div>
      <div style={{ flex: 1, marginBottom: 24 }}>
        <img src={articulo} alt="Top Articles" style={{ width: "100%" }} />
      </div>
    </div>
  ) : null)}
  </>
  )}

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
