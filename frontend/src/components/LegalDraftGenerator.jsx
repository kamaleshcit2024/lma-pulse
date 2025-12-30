import React, { useState, useEffect } from 'react';
import { FileText, Download, Send, Loader } from 'lucide-react';
import apiService from '../services/api';
import './LegalDraftGenerator.css';

const LegalDraftGenerator = () => {
    const [breach, setBreach] = useState(null);
    const [document, setDocument] = useState(null);
    const [generating, setGenerating] = useState(false);
    const [documentType, setDocumentType] = useState('reservation');

    useEffect(() => {
        // Listen for breach events from Dashboard
        const handleGenerateEvent = (event) => {
            setBreach(event.detail);
            generateDocument(event.detail, 'reservation');
        };

        window.addEventListener('generate-legal-doc', handleGenerateEvent);
        return () => window.removeEventListener('generate-legal-doc', handleGenerateEvent);
    }, []);

    const generateDocument = async (breachData, type) => {
        setGenerating(true);
        setDocumentType(type);

        try {
            const response = type === 'reservation'
                ? await apiService.generateReservationOfRights(breachData)
                : await apiService.generateWaiverTemplate(breachData);

            setDocument(response.data);
        } catch (error) {
            console.error('Document generation error:', error);
            alert('Failed to generate document. Please check your OpenAI API key.');
        } finally {
            setGenerating(false);
        }
    };

    const handleDownload = () => {
        if (!document) return;

        const blob = new Blob([document.generated_text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${document.document_type}_${new Date().toISOString().split('T')[0]}.txt`;
        a.click();
    };

    if (!breach && !document) {
        return (
            <div className="legal-generator-empty">
                <FileText size={48} color="#64748b" />
                <p>Select a breached covenant to generate legal documentation</p>
            </div>
        );
    }

    return (
        <div className="legal-generator">
            <div className="generator-header">
                <h2>AI Legal Draft Generator</h2>
                <div className="generator-controls">
                    <button
                        className={`doc-type-btn ${documentType === 'reservation' ? 'active' : ''}`}
                        onClick={() => breach && generateDocument(breach, 'reservation')}
                        disabled={generating}
                    >
                        Reservation of Rights
                    </button>
                    <button
                        className={`doc-type-btn ${documentType === 'waiver' ? 'active' : ''}`}
                        onClick={() => breach && generateDocument(breach, 'waiver')}
                        disabled={generating}
                    >
                        Waiver Template
                    </button>
                </div>
            </div>

            {generating && (
                <div className="generating-overlay">
                    <Loader className="spinner-icon" size={32} />
                    <p>Generating LMA-compliant document with GPT-4o...</p>
                </div>
            )}

            {document && !generating && (
                <>
                    <div className="document-preview">
                        <div className="document-metadata">
                            <span className="metadata-item">
                                <strong>Type:</strong> {document.document_type.replace('_', ' ').toUpperCase()}
                            </span>
                            <span className="metadata-item">
                                <strong>Generated:</strong> {new Date(document.created_at).toLocaleString()}
                            </span>
                            <span className="metadata-item">
                                <strong>Clauses:</strong> {document.clause_references.join(', ')}
                            </span>
                        </div>

                        <div className="document-text">
                            <pre>{document.generated_text}</pre>
                        </div>
                    </div>

                    <div className="document-actions">
                        <button className="action-btn download" onClick={handleDownload}>
                            <Download size={18} />
                            Download Document
                        </button>
                        <button className="action-btn send">
                            <Send size={18} />
                            Send to Borrower
                        </button>
                        <button className="action-btn secondary">
                            <FileText size={18} />
                            Save to DMS
                        </button>
                    </div>
                </>
            )}
        </div>
    );
};

export default LegalDraftGenerator;
