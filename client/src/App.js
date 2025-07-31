import React, { useState } from 'react';
import './App.css'; // Assuming the CSS is in App.css

export default function App() {
  const [headline, setHeadline] = useState('');
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const apiUrl = 'http://127.0.0.1:5000/predict';

  const handleAnalyzeClick = async () => {
    if (!headline.trim()) {
      alert('Please enter a headline.');
      return;
    }

    setIsLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ headline: headline }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();

      if (parseFloat(data.score) < 0.5) {
        data.label = 'Real News';
      }

      setResult(data);
    } catch (err) {
      console.error('Fetch error:', err);
      setError('Could not connect to the model. Please ensure the Python server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const ResultCard = ({ data }) => {
    if (!data) return null;

    const { label, score } = data;
    const isFake = label === 'Fake News';
    const resultClass = isFake ? 'result-fake' : 'result-real';

    return (
      <div className={`result-card ${resultClass}`}>
        <div className="result-header">
          <h3>Prediction Result</h3>
          <div className="result-label">{label}</div>
        </div>
        <p>
          The model classified this headline as <strong>{label}</strong> with a confidence score of <strong>{score}</strong>.
        </p>
        <p className="note">
          Note: A higher score indicates greater confidence in the prediction.
        </p>
      </div>
    );
  };

  return (
    <div className="app-container">
      <div className="app-box">
        <h1>Fake News Detector</h1>
        <p className="subtext">Enter a headline to get a prediction from the AI model.</p>

        <textarea
          rows="4"
          className="headline-input"
          placeholder="e.g., Trump officials brief Hill staff on Saudi reactors..."
          value={headline}
          onChange={(e) => setHeadline(e.target.value)}
          disabled={isLoading}
        />

        <button
          className="analyze-button"
          onClick={handleAnalyzeClick}
          disabled={isLoading}
        >
          {isLoading ? 'Analyzing...' : 'Analyze Headline'}
        </button>

        <div className="result-container">
          {isLoading && <div className="loading">Loading result...</div>}
          {error && <div className="error">{error}</div>}
          {result && <ResultCard data={result} />}
        </div>
      </div>
    </div>
  );
}
