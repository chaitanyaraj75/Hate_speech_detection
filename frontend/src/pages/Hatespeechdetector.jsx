import React, { useState } from "react";

const HateSpeechDetector = () => {
  const [tweet, setTweet] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      const response = await fetch("http://127.0.0.1:3001/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ tweet }),
      });

      if (!response.ok) {
        const txt = await response.text().catch(() => response.statusText);
        throw new Error(`Server error: ${response.status} ${txt}`);
      }

      const data = await response.json();
      console.log("Prediction response:", response, data);
      setPrediction(data.prediction);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const clear = () => {
    setTweet("");
    setPrediction(null);
    setError("");
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2">
          <div className="card bg-base-100 shadow-md">
            <div className="card-body">
              <h2 className="card-title">Hate Speech Detector</h2>
              <p className="text-sm text-gray-500">Paste a tweet or sentence below to check whether it contains hate speech.</p>

              <form onSubmit={handleSubmit} className="mt-4">
                <textarea
                  rows={8}
                  placeholder="Enter text to analyze..."
                  value={tweet}
                  onChange={(e) => setTweet(e.target.value)}
                  required
                  className="textarea textarea-bordered w-full resize-none"
                />

                <div className="flex flex-col sm:flex-row gap-3 mt-4">
                  <button type="submit" className="btn btn-primary" disabled={loading}>
                    {loading ? (
                      <span className="loading loading-spinner" />
                    ) : (
                      'Predict'
                    )}
                  </button>
                  <button type="button" onClick={clear} className="btn btn-ghost">Clear</button>
                  <button type="button" onClick={() => setTweet('I hate everyone')} className="btn btn-outline">Example</button>
                </div>
              </form>

              {error && (
                <div className="alert alert-error mt-4">
                  <div>
                    <span>{error}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div>
          <div className="card bg-base-100 shadow-md sticky top-24">
            <div className="card-body">
              <h3 className="font-semibold text-lg">Result</h3>
              <div className="mt-3">
                {loading && <div className="text-sm text-gray-500">Analyzing...</div>}

                {!loading && prediction === null && !error && (
                  <div className="text-sm text-gray-500">No analysis yet. Submit text to see results.</div>
                )}

                {!loading && prediction !== null && (
                  <div className="mt-2">
                    {prediction === 1 ? (
                      <div className="badge badge-error gap-2 text-lg">Hate Speech 🚨</div>
                    ) : (
                      <div className="badge badge-success gap-2 text-lg">Not Hate Speech ✅</div>
                    )}
                    <div className="mt-3">
                      <h4 className="font-medium">Raw prediction:</h4>
                      <pre className="bg-gray-100 p-2 rounded mt-2">{String(prediction)}</pre>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HateSpeechDetector;
