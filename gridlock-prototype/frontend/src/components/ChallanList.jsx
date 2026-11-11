import React, { useCallback, useEffect, useState } from "react";
import { FileDown, Loader2, Trash2 } from "lucide-react";
import { fetchChallans, downloadChallanPdf, deleteAllChallans } from "../api/client";
import "./ChallanList.css";

export default function ChallanList() {
  const [challans, setChallans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [downloadingId, setDownloadingId] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);

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

  async function handleDeleteAll() {
    try {
      setDeleting(true);
      setError("");
      await deleteAllChallans();
      setChallans([]);
    } catch (err) {
      setError(err.message || "Delete failed.");
    } finally {
      setDeleting(false);
      setConfirming(false);
    }
  }

  return (
    <section className="challan-section">
      <div className="challan-head">
        <div>
          <h2 className="challan-title">E-Challan Records</h2>
          <p className="challan-sub">
            Saved traffic violations and downloadable draft challans
          </p>
        </div>

        {!confirming ? (
          <button
            className="btn-danger"
            disabled={challans.length === 0}
            onClick={() => setConfirming(true)}
          >
            <Trash2 size={14} /> Delete All
          </button>
        ) : (
          <div className="confirm-box">
            <span>Delete all {challans.length} challans?</span>
            <button className="btn-danger" disabled={deleting} onClick={handleDeleteAll}>
              {deleting ? <Loader2 size={14} className="spin" /> : "Yes, delete"}
            </button>
            <button className="btn-ghost" onClick={() => setConfirming(false)}>
              Cancel
            </button>
          </div>
        )}
      </div>

      {error && <p role="alert" className="challan-error">{error}</p>}

      {loading ? (
        <p className="challan-msg">Loading challans...</p>
      ) : challans.length === 0 ? (
        <p className="challan-msg">No challans found yet. Detect a traffic violation first.</p>
      ) : (
        <>
          <p className="challan-count">
            Total challans: <strong>{challans.length}</strong>
          </p>

          <div className="challan-table-wrap">
            <table className="challan-table">
              <thead>
                <tr>
                  <th>Challan ID</th>
                  <th>Plate Number</th>
                  <th>Violation</th>
                  <th>Time</th>
                  <th>Status</th>
                  <th>PDF</th>
                </tr>
              </thead>
              <tbody>
                {challans.map((c) => {
                  const busy = downloadingId === c.challan_id;
                  return (
                    <tr key={c.challan_id}>
                      <td className="mono">{c.challan_id}</td>
                      <td>
                        <span className="plate-tag mono">{c.plate_number || "UNKNOWN"}</span>
                      </td>
                      <td>
                        <span className={`type-pill pill-${(c.violation_type || "").toLowerCase()}`}>
                          {(c.violation_type || "").replace(/_/g, " ")}
                        </span>
                      </td>
                      <td className="mono">{c.violation_time || "N/A"}</td>
                      <td>
                        <span className={`status-pill ${c.status === "DRAFT" ? "st-draft" : "st-other"}`}>
                          {c.status}
                        </span>
                      </td>
                      <td>
                        <button
                          className="btn-pdf"
                          disabled={busy}
                          onClick={() => handleDownload(c.challan_id)}
                        >
                          {busy ? <Loader2 size={14} className="spin" /> : <FileDown size={14} />}
                          {busy ? "Downloading" : "PDF"}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}