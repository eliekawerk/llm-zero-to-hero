// Main App component
const App = () => {
  const [data, setData] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [currentIndex, setCurrentIndex] = React.useState(0);
  const [critiques, setCritiques] = React.useState({});
  const [pdfUrl, setPdfUrl] = React.useState(null);
  const [jsonFile, setJsonFile] = React.useState("./data/evaluation_report.json");
  const [availableFiles, setAvailableFiles] = React.useState([]);

  // Fetch available JSON files
  React.useEffect(() => {
    fetch(`./api/list-json-files`)
      .then(response => {
        if (!response.ok) {
          // Silently fail - we'll just use the default file
          console.warn("Could not fetch available JSON files");
          return { files: [] };
        }
        return response.json();
      })
      .then(data => {
        if (data.files && Array.isArray(data.files)) {
          setAvailableFiles(data.files);
        }
      })
      .catch(err => {
        console.warn("Error fetching available JSON files:", err);
      });
  }, []);

  // Load the selected JSON file
  const loadData = React.useCallback(() => {
    if (!jsonFile) return;
    
    setLoading(true);
    setError(null);
    
    const timestamp = new Date().getTime();
    // Fix the path by ensuring it's properly resolved relative to the server
    const filePath = jsonFile.replace('../data/', './data/');
    
    fetch(`${filePath}?t=${timestamp}`)
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
  }, [jsonFile]);

  // Load data when component mounts or jsonFile changes
  React.useEffect(() => {
    loadData();
  }, [jsonFile, loadData]);

  // Handle file selection change
  const handleFileChange = (e) => {
    setJsonFile(e.target.value);
  };

  // Update PDF URL when current index changes
  React.useEffect(() => {
    if (data && data.detailed_results && data.detailed_results[currentIndex]) {
      const resumeName = data.detailed_results[currentIndex].resume;
      if (resumeName) {
        // Extract the base file name from the resume path if needed
        const fileName = resumeName.includes('/') 
          ? resumeName.split('/').pop() 
          : resumeName;
          
        // Check if the file has an extension, if not, add .pdf
        const pdfFileName = fileName.includes('.') ? fileName : `${fileName}.pdf`;
        
        // Create URL to the PDF file in the resumes folder - fix path for server
        setPdfUrl(`./resumes/${pdfFileName}`);
      } else {
        setPdfUrl(null);
      }
    }
  }, [currentIndex, data]);

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

  return (
    <div className="container">
      <Header 
        exportToJSON={exportToJSON} 
        exportToCSV={exportToCSV}
        jsonFile={jsonFile}
        availableFiles={availableFiles}
        onFileChange={handleFileChange}
        onRefresh={loadData}
      />
      
      {loading ? (
        <div className="loading">Loading evaluation data...</div>
      ) : error ? (
        <div className="error">Error loading data: {error}</div>
      ) : !data || !data.detailed_results || data.detailed_results.length === 0 ? (
        <div className="no-data">No evaluation data found.</div>
      ) : (
        <React.Fragment>
          <Navigation 
            currentIndex={currentIndex} 
            totalResults={data.detailed_results.length} 
            goToPrevious={goToPrevious} 
            goToNext={goToNext} 
          />
          
          <div className="keyboard-help">
            <small>Tip: Use left/right arrow keys to navigate between records</small>
          </div>
          
          {/* PDF Viewer positioned between navigation and results */}
          <PdfViewer pdfUrl={pdfUrl} />
          
          <ResultCard 
            result={data.detailed_results[currentIndex]} 
            critique={critiques[currentIndex] || ""} 
            onCritiqueChange={handleCritiqueChange} 
          />
        </React.Fragment>
      )}
    </div>
  );
};

// Header Component
const Header = ({ exportToJSON, exportToCSV, jsonFile, availableFiles, onFileChange }) => {
  return (
    <header>
      <h1>Evaluation Results Annotator</h1>
      <nav className="main-nav">
        <a href="error_analysis.html" className="nav-link">Error Analysis Dashboard</a>
        <a href="evaluation_report.html" className="nav-link">Evaluation Report</a>
      </nav>
      <div className="export-btns">
        <button className="btn" onClick={exportToJSON}>Export to JSON</button>
        <button className="btn btn-secondary" onClick={exportToCSV}>Export to CSV</button>
      </div>
      <div className="file-selector">
        <select value={jsonFile} onChange={onFileChange}>
          {availableFiles.map(file => (
            <option key={file} value={file}>{file}</option>
          ))}
        </select>
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
  // Initialize with existing user critique if available
  React.useEffect(() => {
    if (result.user_critique && result.user_critique.length > 0 && !critique) {
      onCritiqueChange({ target: { value: result.user_critique } });
    }
  }, [result, critique, onCritiqueChange]);

  // Format percentage_correct as a percentage if it exists
  const formattedPercentage = result.percentage_correct !== undefined 
    ? `${Math.round(result.percentage_correct * 100)}%` 
    : 'N/A';

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

      <div className="result-metrics">
        <span className={`result-status ${result.is_correct ? 'correct' : 'incorrect'}`}>
          {result.is_correct ? 'CORRECT' : 'INCORRECT'}
        </span>
        
        <span className="percentage-correct">
          <span className="percentage-label">Correctness:</span>
          <span className={`percentage-value ${
            result.percentage_correct >= 0.8 ? 'high' : 
            result.percentage_correct >= 0.5 ? 'medium' : 'low'
          }`}>
            {formattedPercentage}
          </span>
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

// PDF Viewer Component
const PdfViewer = ({ pdfUrl }) => {
  const canvasRef = React.useRef(null);
  const [pdfDoc, setPdfDoc] = React.useState(null);
  const [pageNum, setPageNum] = React.useState(1);
  const [totalPages, setTotalPages] = React.useState(0);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [expanded, setExpanded] = React.useState(false);

  // Toggle expanded state
  const toggleExpand = () => {
    setExpanded(!expanded);
  };

  // Load PDF when URL changes
  React.useEffect(() => {
    if (!pdfUrl || !expanded) {
      setPdfDoc(null);
      setTotalPages(0);
      return;
    }

    setLoading(true);
    setError(null);

    // Use PDF.js to load the PDF
    const loadPdf = async () => {
      try {
        const pdfjsLib = window['pdfjs-dist/build/pdf'];
        pdfjsLib.GlobalWorkerOptions.workerSrc = window['pdfjs-dist/build/pdf.worker'];
        
        const pdf = await pdfjsLib.getDocument(pdfUrl).promise;
        setPdfDoc(pdf);
        setTotalPages(pdf.numPages);
        setPageNum(1); // Reset to first page when loading a new PDF
        setLoading(false);
      } catch (err) {
        console.error('Error loading PDF:', err);
        setError(`Failed to load PDF: ${err.message}`);
        setLoading(false);
      }
    };

    loadPdf();
  }, [pdfUrl, expanded]);

  // Render PDF page when page number changes or PDF document changes
  React.useEffect(() => {
    if (!pdfDoc || !canvasRef.current || !expanded) return;

    const renderPage = async () => {
      try {
        const page = await pdfDoc.getPage(pageNum);
        
        // Get container width to calculate appropriate scale
        const container = canvasRef.current.parentElement;
        const containerWidth = container.clientWidth - 20; // Account for padding
        
        // Get viewport at scale 1.0 first to calculate dimensions
        const originalViewport = page.getViewport({ scale: 1.0 });
        
        // Calculate scale to fit width while maintaining aspect ratio
        const scale = containerWidth / originalViewport.width;
        
        // Create viewport with the calculated scale
        const viewport = page.getViewport({ scale });
        
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        
        canvas.height = viewport.height;
        canvas.width = viewport.width;
        
        const renderContext = {
          canvasContext: context,
          viewport: viewport
        };
        
        await page.render(renderContext).promise;
      } catch (err) {
        console.error('Error rendering PDF page:', err);
        setError(`Failed to render page: ${err.message}`);
      }
    };

    renderPage();
  }, [pdfDoc, pageNum, expanded]);

  // Handle page navigation
  const goToPreviousPage = () => {
    if (pageNum > 1) {
      setPageNum(pageNum - 1);
    }
  };

  const goToNextPage = () => {
    if (pageNum < totalPages) {
      setPageNum(pageNum + 1);
    }
  };

  // If no PDF URL
  if (!pdfUrl) {
    return (
      <div className="pdf-viewer pdf-toggle-container">
        <div className="pdf-toggle">
          <div className="pdf-toggle-btn disabled">
            <span>No PDF available for this record</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="pdf-viewer">
      <div className="pdf-toggle">
        <button 
          className="pdf-toggle-btn" 
          onClick={toggleExpand}
        >
          {expanded ? 'Hide PDF' : 'Show PDF'} 
          <span className={`toggle-arrow ${expanded ? 'up' : 'down'}`}>
            {expanded ? '▲' : '▼'}
          </span>
        </button>
        {expanded && <span className="pdf-filename">{pdfUrl.split('/').pop()}</span>}
      </div>
      
      <div className={`pdf-content ${expanded ? 'expanded' : 'collapsed'}`}>
        {expanded && (
          <React.Fragment>
            {loading ? (
              <div className="pdf-placeholder">
                <div>Loading PDF...</div>
              </div>
            ) : error ? (
              <div className="pdf-placeholder">
                <div className="error">{error}</div>
              </div>
            ) : (
              <React.Fragment>
                <div className="pdf-controls">
                  <button 
                    className="btn pdf-nav-btn" 
                    onClick={goToPreviousPage} 
                    disabled={pageNum <= 1}
                  >
                    &larr; Previous Page
                  </button>
                  <div className="pdf-page-info">
                    Page {pageNum} of {totalPages}
                  </div>
                  <button 
                    className="btn pdf-nav-btn" 
                    onClick={goToNextPage} 
                    disabled={pageNum >= totalPages}
                  >
                    Next Page &rarr;
                  </button>
                </div>
                <div className="pdf-container">
                  <canvas ref={canvasRef} className="pdf-canvas"></canvas>
                </div>
              </React.Fragment>
            )}
          </React.Fragment>
        )}
      </div>
    </div>
  );
};

// Render the app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);