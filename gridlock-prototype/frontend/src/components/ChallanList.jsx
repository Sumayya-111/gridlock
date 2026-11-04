
import React, { useCallback, useEffect, useState } from "react";
import { fetchChallans, downloadChallanPdf } from "../api/client";

export default function ChallanList() {
  const [challans, setChallans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [downloadingId, setDownloadingId] = useState("");

  const loadChallans = useCallback(async () => {
    try {
      setError("");
      const data = await fetchChallans();
      setChallans(data.records || []);
    } catch (err) {
      setError(err.message || "Could not load challans.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadChallans();
  }, [loadChallans]);

  async function handleDownload(challanId) {
    try {
      setDownloadingId(challanId);
      setError("");
      await downloadChallanPdf(challanId);
    } catch (err) {
      setError(err.message || "PDF download failed.");
    } finally {
      setDownloadingId("");
    }
  }

  const cellStyle = {
    padding: "12px",
    borderBottom: "1px solid #e2e8f0",
    textAlign: "left",
    fontSize: "13px",
    whiteSpace: "nowrap",
  };

  const buttonStyle = {
    padding: "8px 12px",
    border: "none",
    borderRadius: "6px",
    background: "#0f766e",
    color: "#ffffff",
    cursor: "pointer",
    fontWeight: 600,
  };

  return (
    <section style={{ padding: "20px", color: "inherit" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "12px",
          flexWrap: "wrap",
          marginBottom: "16px",
        }}
      >
        <div>
          <h2 style={{ margin: 0 }}>E-Challan Records</h2>
          <p style={{ marginTop: "6px", opacity: 0.7 }}>
            Saved traffic violations and downloadable draft challans
          </p>
        </div>

        <button style={buttonStyle} onClick={loadChallans}>
          Refresh
        </button>
      </div>

      {error && (
        <p role="alert" style={{ color: "#dc2626" }}>
          {error}
        </p>
      )}

      {loading ? (
        <p>Loading challans...</p>
      ) : challans.length === 0 ? (
        <p>No challans found yet. Detect a traffic violation first.</p>
      ) : (
        <>
          <p style={{ opacity: 0.75 }}>
            Total challans: <strong>{challans.length}</strong>
          </p>

          <div style={{ overflowX: "auto", width: "100%" }}>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                minWidth: "750px",
              }}
            >
              <thead>
                <tr style={{ background: "rgba(148, 163, 184, 0.12)" }}>
                  <th style={cellStyle}>Challan ID</th>
                  <th style={cellStyle}>Plate Number</th>
                  <th style={cellStyle}>Violation</th>
                  <th style={cellStyle}>Time</th>
                  <th style={cellStyle}>Status</th>
                  <th style={cellStyle}>PDF</th>
                </tr>
              </thead>

              <tbody>
                {challans.map((challan) => (
                  <tr key={challan.challan_id}>
                    <td style={cellStyle}>{challan.challan_id}</td>
                    <td style={cellStyle}>
                      {challan.plate_number || "UNKNOWN"}
                    </td>
                    <td style={cellStyle}>{challan.violation_type}</td>
                    <td style={cellStyle}>
                      {challan.violation_time || "N/A"}
                    </td>
                    <td style={cellStyle}>
                      <span
                        style={{
                          fontWeight: 600,
                          color:
                            challan.status === "DRAFT"
                              ? "#0f766e"
                              : "#b45309",
                        }}
                      >
                        {challan.status}
                      </span>
                    </td>
                    <td style={cellStyle}>
                      <button
                        style={{
                          ...buttonStyle,
                          opacity:
                            downloadingId === challan.challan_id ? 0.6 : 1,
                        }}
                        disabled={downloadingId === challan.challan_id}
                        onClick={() =>
                          handleDownload(challan.challan_id)
                        }
                      >
                        {downloadingId === challan.challan_id
                          ? "Downloading..."
                          : "Download PDF"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}
