import React, { useState } from 'react';
import { Modal, Button, Alert, Spinner } from 'react-bootstrap';

const DocumentViewer = ({ show, onHide, fileUrl, fileName, fileType }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const handleClose = () => {
    setLoading(true);
    setError(null);
    onHide();
  };

  const getFileExtension = (filename) => {
    return filename?.split('.').pop()?.toLowerCase() || '';
  };

  const renderDocumentContent = () => {
    const extension = getFileExtension(fileName);

    // For PDF files
    if (extension === 'pdf' || fileType === 'application/pdf') {
      return (
        <div className="text-center">
          {loading && (
            <div className="position-absolute w-100 h-100 d-flex justify-content-center align-items-center bg-light">
              <div>
                <Spinner animation="border" role="status">
                  <span className="visually-hidden">Loading...</span>
                </Spinner>
                <p className="mt-2">Loading PDF...</p>
              </div>
            </div>
          )}
          <iframe
            src={`${fileUrl}#toolbar=1&navpanes=1&scrollbar=1`}
            style={{ 
              width: '100%', 
              height: '70vh', 
              border: '1px solid #ddd',
              borderRadius: '4px'
            }}
            title={fileName}
            onLoad={() => setLoading(false)}
            onError={() => {
              setError('Failed to load PDF. Your browser may not support PDF viewing.');
              setLoading(false);
            }}
          />
        </div>
      );
    }

    // For image files
    if (['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg'].includes(extension) || 
        fileType?.startsWith('image/')) {
      return (
        <div className="text-center">
          {loading && (
            <div className="d-flex justify-content-center align-items-center" style={{ height: '200px' }}>
              <Spinner animation="border" />
            </div>
          )}
          <img 
            src={fileUrl} 
            alt={fileName}
            style={{ 
              maxWidth: '100%', 
              maxHeight: '70vh',
              display: loading ? 'none' : 'block',
              margin: '0 auto'
            }}
            onLoad={() => setLoading(false)}
            onError={() => {
              setError('Failed to load image.');
              setLoading(false);
            }}
          />
        </div>
      );
    }

    // For text files
    if (['txt', 'md', 'json', 'xml', 'csv', 'js', 'css', 'html'].includes(extension) || 
        fileType?.startsWith('text/')) {
      return (
        <div>
          {loading && (
            <div className="text-center p-4">
              <Spinner animation="border" />
              <p className="mt-2">Loading document...</p>
            </div>
          )}
          <iframe
            src={fileUrl}
            style={{ 
              width: '100%', 
              height: '70vh', 
              border: '1px solid #ddd',
              borderRadius: '4px',
              display: loading ? 'none' : 'block'
            }}
            title={fileName}
            onLoad={() => setLoading(false)}
            onError={() => {
              setError('Failed to load text file.');
              setLoading(false);
            }}
          />
        </div>
      );
    }

    // For Word documents, Excel files, PowerPoint
    if (['doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx'].includes(extension)) {
      return (
        <div>
          <Alert variant="info" className="mb-3">
            <Alert.Heading>Office Document</Alert.Heading>
            <p>This is a Microsoft Office document. Choose how you'd like to view it:</p>
          </Alert>
          
          <div className="d-flex gap-3 justify-content-center">
            <Button 
              variant="primary" 
              href={`https://view.officeapps.live.com/op/view.aspx?src=${encodeURIComponent(fileUrl)}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              📄 View in Office Online
            </Button>
            
            <Button 
              variant="outline-primary"
              href={`https://docs.google.com/gview?url=${encodeURIComponent(fileUrl)}&embedded=true`}
              target="_blank"
              rel="noopener noreferrer"
            >
              📑 View in Google Docs Viewer
            </Button>
          </div>

          <div className="mt-4">
            <Alert variant="secondary">
              <strong>Alternative:</strong> You can also download the file and open it with Microsoft Office or compatible software.
            </Alert>
          </div>
        </div>
      );
    }

    // For video files
    if (['mp4', 'webm', 'ogg', 'avi', 'mov'].includes(extension) || 
        fileType?.startsWith('video/')) {
      return (
        <div className="text-center">
          <video 
            controls 
            style={{ maxWidth: '100%', maxHeight: '70vh' }}
            onLoadStart={() => setLoading(false)}
            onError={() => {
              setError('Failed to load video.');
              setLoading(false);
            }}
          >
            <source src={fileUrl} type={fileType || `video/${extension}`} />
            Your browser does not support the video tag.
          </video>
        </div>
      );
    }

    // For audio files
    if (['mp3', 'wav', 'ogg', 'm4a', 'flac'].includes(extension) || 
        fileType?.startsWith('audio/')) {
      return (
        <div className="text-center p-4">
          <div className="mb-3">
            <h5>🎵 Audio File</h5>
            <p className="text-muted">{fileName}</p>
          </div>
          <audio 
            controls 
            className="w-100"
            onLoadStart={() => setLoading(false)}
            onError={() => {
              setError('Failed to load audio.');
              setLoading(false);
            }}
          >
            <source src={fileUrl} type={fileType || `audio/${extension}`} />
            Your browser does not support the audio element.
          </audio>
        </div>
      );
    }

    // Fallback for other file types
    return (
      <Alert variant="warning">
        <Alert.Heading>Preview Not Available</Alert.Heading>
        <p>Preview is not supported for this file type: <strong>.{extension}</strong></p>
        <p>Please download the file to view it locally.</p>
        <hr />
        <div className="d-flex gap-2">
          <Button variant="primary" href={fileUrl} download={fileName}>
            📥 Download File
          </Button>
          <Button 
            variant="outline-primary"
            onClick={() => window.open(fileUrl, '_blank')}
          >
            🔗 Open in New Tab
          </Button>
        </div>
      </Alert>
    );
  };

  return (
    <Modal show={show} onHide={handleClose} size="xl" centered>
      <Modal.Header closeButton>
        <Modal.Title>
          📄 {fileName}
        </Modal.Title>
      </Modal.Header>
      <Modal.Body style={{ maxHeight: '80vh', overflowY: 'auto', position: 'relative' }}>
        {error ? (
          <Alert variant="danger">
            <Alert.Heading>Cannot Display Document</Alert.Heading>
            <p>{error}</p>
            <hr />
            <div className="d-flex gap-2">
              <Button variant="primary" href={fileUrl} download={fileName}>
                📥 Download File
              </Button>
              <Button 
                variant="outline-primary"
                onClick={() => window.open(fileUrl, '_blank')}
              >
                🔗 Open in New Tab
              </Button>
            </div>
          </Alert>
        ) : (
          renderDocumentContent()
        )}
      </Modal.Body>
      <Modal.Footer className="d-flex justify-content-between">
        <div>
          <small className="text-muted">
            File size and type information available on download
          </small>
        </div>
        <div className="d-flex gap-2">
          <Button variant="secondary" onClick={handleClose}>
            Close
          </Button>
          <Button variant="primary" href={fileUrl} download={fileName}>
            📥 Download
          </Button>
        </div>
      </Modal.Footer>
    </Modal>
  );
};

export default DocumentViewer;
