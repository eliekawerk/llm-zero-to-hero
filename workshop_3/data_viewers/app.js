// Main App component
const App = () => {
  const [data, setData] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);
  const [currentIndex, setCurrentIndex] = React.useState(0);
  const [critiques, setCritiques] = React.useState({});

  // Fetch data from JSON file
  React.useEffect(() => {
    const timestamp = new Date().getTime()
    fetch(`../data/evaluation_report.json?t=${timestamp}`)
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(jsonData => {
        setData(jsonData);
        // Initialize critiques object with empty strings for each result
        const initialCritiques = {};
        jsonData.detailed_results.forEach((_, index) => {
          initialCritiques[index] = "";
        });
        setCritiques(initialCritiques);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  // Handle navigation
  const goToNext = () => {
    if (data && currentIndex < data.detailed_results.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const goToPrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  // Handle keyboard navigation
  React.useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'ArrowRight') {
        goToNext();
      } else if (event.key === 'ArrowLeft') {
        goToPrevious();
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    // Clean up event listener when component unmounts
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [currentIndex, data]); // Re-add event listener when these dependencies change

  // Handle critique changes
  const handleCritiqueChange = (e) => {
    setCritiques({
      ...critiques,
      [currentIndex]: e.target.value
    });
  };

  // Export functions
  const exportToJSON = () => {
    if (!data) return;

    // Add critiques to each detailed result
    const dataWithCritiques = {
      ...data,
      detailed_results: data.detailed_results.map((result, index) => ({
        ...result,
        user_critique: critiques[index] || ""
      }))
    };

    const jsonStr = JSON.stringify(dataWithCritiques, null, 2);
    const dataUri = `data:application/json;charset=utf-8,${encodeURIComponent(jsonStr)}`;
    
    const link = document.createElement('a');
    link.setAttribute('href', dataUri);
    link.setAttribute('download', 'evaluation_report_with_critiques.json');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportToCSV = () => {
    if (!data) return;

    // CSV header
    const headers = [
      'Resume',
      'Question',
      'Gold Answer',
      'Predicted Answer',
      'Is Correct',
      'Reasoning',
      'Missing Info',
      'Incorrect Info',
      'User Critique'
    ];

    // CSV rows
    const rows = data.detailed_results.map((result, index) => [
      result.resume,
      result.question,
      result.gold_answer,
      result.predicted_answer,
      result.is_correct,
      result.reasoning,
      result.missing_info ? result.missing_info.join('; ') : '',
      result.incorrect_info ? result.incorrect_info.join('; ') : '',
      critiques[index] || ''
    ]);

    // Convert to CSV string
    const csvContent = [
      headers.join(','),
      ...rows.map(row => 
        row.map(cell => 
          typeof cell === 'string' ? `"${cell.replace(/"/g, '""')}"` : cell
        ).join(',')
      )
    ].join('\n');

    // Create download link
    const dataUri = `data:text/csv;charset=utf-8,${encodeURIComponent(csvContent)}`;
    const link = document.createElement('a');
    link.setAttribute('href', dataUri);
    link.setAttribute('download', 'evaluation_report_with_critiques.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (loading) {
    return <div className="container">Loading evaluation data...</div>;
  }

  if (error) {
    return <div className="container">Error loading data: {error}</div>;
  }

  if (!data || !data.detailed_results || data.detailed_results.length === 0) {
    return <div className="container">No evaluation data found.</div>;
  }

  const currentResult = data.detailed_results[currentIndex];

  return (
    <div className="container">
      <Header 
        exportToJSON={exportToJSON} 
        exportToCSV={exportToCSV} 
      />
      
      <Navigation 
        currentIndex={currentIndex} 
        totalResults={data.detailed_results.length} 
        goToPrevious={goToPrevious} 
        goToNext={goToNext} 
      />
      
      <div className="keyboard-help">
        <small>Tip: Use left/right arrow keys to navigate between records</small>
      </div>
      
      <ResultCard 
        result={currentResult} 
        critique={critiques[currentIndex] || ""} 
        onCritiqueChange={handleCritiqueChange} 
      />
    </div>
  );
};

// Header Component
const Header = ({ exportToJSON, exportToCSV }) => {
  return (
    <header>
      <h1>Evaluation Results Annotator</h1>
      <div className="export-btns">
        <button className="btn" onClick={exportToJSON}>Export to JSON</button>
        <button className="btn btn-secondary" onClick={exportToCSV}>Export to CSV</button>
      </div>
    </header>
  );
};

// Navigation Component
const Navigation = ({ currentIndex, totalResults, goToPrevious, goToNext }) => {
  return (
    <div className="navigation">
      <div className="nav-buttons">
        <button className="btn" onClick={goToPrevious} disabled={currentIndex === 0}>
          &larr; Previous
        </button>
        <button className="btn" onClick={goToNext} disabled={currentIndex === totalResults - 1}>
          Next &rarr;
        </button>
      </div>
      <div className="progress">
        Result {currentIndex + 1} of {totalResults}
      </div>
    </div>
  );
};

// Result Card Component
const ResultCard = ({ result, critique, onCritiqueChange }) => {
  return (
    <div className="record-card">
      <div className="record-details">
        <div className="field">
          <span className="field-label">Resume:</span>
          <div className="field-value">{result.resume}</div>
        </div>
        <div className="field">
          <span className="field-label">Question:</span>
          <div className="field-value">{result.question}</div>
        </div>
      </div>

      <div className="answer-comparison">
        <div>
          <span className="field-label">Gold Answer:</span>
          <div className="gold-answer">{result.gold_answer}</div>
        </div>
        <div>
          <span className="field-label">Predicted Answer:</span>
          <div className={`predicted-answer ${result.is_correct ? 'correct' : 'incorrect'}`}>
            {result.predicted_answer}
          </div>
        </div>
      </div>

      <div>
        <span className={`result-status ${result.is_correct ? 'correct' : 'incorrect'}`}>
          {result.is_correct ? 'CORRECT' : 'INCORRECT'}
        </span>
      </div>

      <div className="field">
        <span className="field-label">Reasoning:</span>
        <div className="field-value">{result.reasoning}</div>
      </div>

      {result.missing_info && result.missing_info.length > 0 && (
        <div className="errors-section">
          <span className="field-label">Missing Information:</span>
          <ul className="error-list">
            {result.missing_info.map((item, index) => (
              <li key={`missing-${index}`}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      {result.incorrect_info && result.incorrect_info.length > 0 && (
        <div className="errors-section">
          <span className="field-label">Incorrect Information:</span>
          <ul className="error-list">
            {result.incorrect_info.map((item, index) => (
              <li key={`incorrect-${index}`}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="critique-section">
        <span className="field-label">Your Critique:</span>
        <textarea 
          className="critique-textarea" 
          value={critique} 
          onChange={onCritiqueChange} 
          placeholder="Add your critique or notes here..."
        />
      </div>
    </div>
  );
};

// Render the app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);