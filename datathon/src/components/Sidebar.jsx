import ProfilePicture from "../images/profilePicture.png";
import { useState } from "react";

export const Sidebar = ({selected, setSelected}) => {
  const [hovered, setHovered] = useState(null);

  return (
    <div style={styles.sidebarContainer}>
      <div style={styles.centeredContainer}>
        <div style={styles.profileContainer}>
          <img
            src={ProfilePicture}
            style={styles.profileImage}
            alt="Profile"
          />
        </div>
        <h3 style={styles.name}>Mayra de Luna</h3>
        <div style={styles.tabContainer}>
          <div
            onClick={() => setSelected("prediction")}
            style={{
              ...styles.tab,
              backgroundColor: selected === "prediction" ? "#FFF" : "#7D0A0A",
              color: selected === "prediction" ? "#000" : "#FFF",
            }}
            onMouseEnter={() => setHovered("prediction")}
            onMouseLeave={() => setHovered(null)}
          >
            <h4>Predicción</h4>
          </div>
          <div
            onClick={() => setSelected("analisis")}
            style={{
              ...styles.tab,
              borderTop: "none",
              backgroundColor: selected === "analisis" ? "#FFF" : "#7D0A0A",
              color: selected === "analisis" ? "#000" : "#FFF",
            }}
            onMouseEnter={() => setHovered("analisis")}
            onMouseLeave={() => setHovered(null)}
          >
            <h4>Análisis</h4>
          </div>
        </div>
      </div>

      <div style={{...styles.session, 
        backgroundColor: hovered === "session" ? "lightGray" : "#FFFFFF",
      }}
      onMouseEnter={() => setHovered("session")}
      onMouseLeave={() => setHovered(null)}
      >
        <h4>Cerrar Sesión</h4>
      </div>
    </div>
  );
};

const styles = {
  sidebarContainer: {
    position: "fixed",
    width: 352,
    height: "100vh",
    backgroundColor: "#7D0A0A",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "space-between",
    boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
  },
  centeredContainer: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    width: "100%",
    marginTop: 48,
  },
  profileContainer: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#FFFFFF",
    borderRadius: "50%",
    width: 164,
    height: 164,
  },
  profileImage: {
    width: "100%",
    borderRadius: "50%",
  },
  name: {
    textAlign: "center",
    color: "#FFFFFF",
    marginTop: 24,
    fontSize: 24,
    marginBottom: 64,
  },
  tabContainer: {
    display: "flex",
    flexDirection: "column",
    width: "100%",
  },
  session: {
    width: 252,
    textAlign: "center",
    backgroundColor: "#FFFFFF",
    marginBottom: 24,
    padding: 8,
    fontSize: 20,
    borderRadius: 8,
  },
  tab: {
    backgroundColor: "#FFFFFF",
    color: "#000000",
    fontSize: 20,
    padding: 24,
    textAlign: "center",
    cursor: "pointer",
    transition: "all 0.2s ease",
    borderBlock: "1px solid #fff"
  },
};
